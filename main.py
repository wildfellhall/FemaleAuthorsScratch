"""FemaleAuthorsScratch — paste each numbered cell into Colab and run in order.
Select an A100 runtime. The corpus list is defined here; Google Drive is not used.
"""

# %% Cell 1: Dependencies and settings
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.2.0", "tokenizers==0.22.2", "safetensors>=0.4.3,<1"], check=True)

from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import importlib.metadata
from pathlib import Path
import math
import random
import shutil
import time
import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

@dataclass
class TrainConfig:
    vocab_size: int = 32768
    min_frequency: int = 2
    context_length: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation: int = 8
    epochs: int = 3
    max_steps: int = 0  # zero: derive the schedule from epochs and actual data size
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    warmup_fraction: float = 0.03
    max_grad_norm: float = 1.0
    eval_every: int = 200
    save_every: int = 100
    log_every: int = 10
    keep_checkpoints: int = 2  # best checkpoint is also retained
    gradient_checkpointing: bool = True
    seed: int = 42
    sample_every: int = 100  # optimizer updates, not accumulation microbatches; 0 disables
    sample_max_new_tokens: int = 128
    sample_temperature: float = 0.8
    sample_top_p: float = 0.95
    sample_seed: int = 1234
    sample_prompts: list[str] = field(default_factory=lambda: [
        "The evening was quiet, and",
        "She opened the letter and",
        '\"I never expected,\" she said,',
    ])

    def validate(self):
        integers = ("vocab_size", "min_frequency", "context_length", "n_layer", "n_head", "n_embd", "batch_size", "gradient_accumulation", "epochs", "eval_every", "save_every", "log_every", "keep_checkpoints")
        if any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in integers):
            raise ValueError("Model dimensions, batch sizes, epochs and intervals must be positive integers.")
        if self.vocab_size < 258 or self.context_length < 8:
            raise ValueError("Byte BPE needs at least 258 tokens; context_length must be >= 8.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if type(self.max_steps) is not int or self.max_steps < 0 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("max_steps must be a nonnegative integer and seed a nonnegative integer.")
        if not 0 <= self.dropout < 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("dropout and warmup_fraction must be in [0, 1).")
        if not all(math.isfinite(x) for x in (self.learning_rate, self.weight_decay, self.max_grad_norm)) or self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer configuration.")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean.")
        if type(self.sample_every) is not int or self.sample_every < 0 or type(self.sample_seed) is not int or self.sample_seed < 0:
            raise ValueError("sample_every and sample_seed must be nonnegative integers.")
        if type(self.sample_max_new_tokens) is not int or self.sample_max_new_tokens < 1:
            raise ValueError("sample_max_new_tokens must be a positive integer.")
        if not math.isfinite(self.sample_temperature) or self.sample_temperature <= 0 or not 0 < self.sample_top_p <= 1:
            raise ValueError("sample_temperature must be finite and positive; sample_top_p must be in (0, 1].")
        if not isinstance(self.sample_prompts, list) or not self.sample_prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in self.sample_prompts):
            raise ValueError("sample_prompts must be a nonempty list of nonempty strings.")


config = TrainConfig()               # Edit the defaults above, or override them here.
OUTPUT_DIR = Path("female-authors")   # Automatically created in Colab's local runtime.
RESUME = False                       # True resumes the latest local checkpoint.
ALLOW_PARTIAL_CORPUS = False         # Review flagged books before enabling.
MAX_SESSION_SECONDS = 7200           # 0 trains until the schedule finishes.
DEVICE = "cuda"                      # "cpu" is for tiny development tests only.
config.validate()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Cell 2: Corpus list and cleaning rules
# Append URLs freely. Related volumes/editions should share an explicit work_id.
sources = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",   #Pride and Prejudice by Jane Austen - written anonymously
    "https://www.gutenberg.org/files/3268/3268-0.txt",   #The Mysteries of Udolpho by Ann Ward Radcliffe - seminal work often referenced in women's novels of the 19th century
    "https://www.gutenberg.org/files/21839/21839-0.txt", #Sense and Sensibility by Jane Austen - written anonymously
    "https://www.gutenberg.org/files/121/121-0.txt",     #Northanger Abbey by Jane Austen - written anonymously
    "https://www.gutenberg.org/files/158/158-0.txt",     #Emma by Jane Austen - written anonymously
    "https://www.gutenberg.org/files/105/105-0.txt",     #Persuasion by Jane Austen - written anonymously
    "https://www.gutenberg.org/files/141/141-0.txt",     #Mansfield Park by Jane Austen - written anonymously
    "https://www.gutenberg.org/files/145/145-0.txt",     #Middlemarch by George Eliot - pseudonym used (real name: Mary Ann Evans)
    "https://www.gutenberg.org/files/550/550-0.txt",     #Silas Marner by George Eliot - pseudonym used (real name: Mary Ann Evans)
    "https://www.gutenberg.org/files/507/507-0.txt",     #Adam Bede by George Eliot - pseudonym used (real name: Mary Ann Evans)
    "https://www.gutenberg.org/files/6688/6688-0.txt",   #The Mill on the Floss by George Eliot - pseudonym used (real name: Mary Ann Evans)
    "https://www.gutenberg.org/files/7469/7469-0.txt",   #Daniel Deronda by George Eliot - pseudonym used (real name: Mary Ann Evans)
    "https://www.gutenberg.org/files/24020/24020-0.txt", #Romola by George Eliot - pseudonym used (real name: Mary Ann Evans)
    "https://www.gutenberg.org/files/969/969-0.txt",     #The Tenant of Wildfell Hall by Anne Brontë
    "https://www.gutenberg.org/files/767/767-0.txt",     #Agnes Grey by Anne Brontë
    "https://www.gutenberg.org/files/1028/1028-0.txt",   #The Professor by Charlotte Brontë
    "https://www.gutenberg.org/files/30486/30486-0.txt", #Shirley by Charlotte Brontë
    "https://www.gutenberg.org/files/9182/9182-0.txt",   #Villette by Charlotte Brontë
    "https://www.gutenberg.org/files/1260/1260-0.txt",   #Jane Eyre: An Autobiography by Charlotte Brontë
    "https://www.gutenberg.org/files/768/768-0.txt",     #Wuthering Heights by Emily Brontë
    "https://www.gutenberg.org/files/37106/37106-0.txt", #Little Women; Or, Meg, Jo, Beth, and Amy by Louisa May Alcott
    "https://www.gutenberg.org/files/28203/28203-0.txt", #Moods by Louisa May Alcott
    "https://www.gutenberg.org/files/41127/41127-0.txt", #Rose in Bloom by Louisa May Alcott
    "https://www.gutenberg.org/files/2787/2787-0.txt",   #An Old-Fashioned Girl by Louisa May Alcott
    "https://www.gutenberg.org/files/2726/2726-0.txt",   #Eight Cousins by Louisa May Alcott
    "https://www.gutenberg.org/files/2786/2786-0.txt",   #Jack and Jill by Louisa May Alcott
    "https://www.gutenberg.org/files/2788/2788-0.txt",   #Little Men: Life at Plumfield With Jo's Boys by Louisa May Alcott
    "https://www.gutenberg.org/files/3499/3499-0.txt",   #Jo's Boys by Louisa May Alcott
    "https://www.gutenberg.org/files/84/84-0.txt",       #Frankenstein; or, the modern prometheus by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/18247/18247-0.txt", #The Last Man by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/15238/15238-0.txt", #Mathilda by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/64329/64329-0.txt", #Falkner: A Novel by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/394/394-0.txt",     #Cranford by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4276/4276-0.txt",   #North and South by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2153/2153-0.txt",   #Mary Barton by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4274/4274-0.txt",   #Wives and Daughters by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4275/4275-0.txt",   #Ruth by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2524/2524-0.txt",   #My Lady Ludlow by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4537/4537-0.txt",   #Sylvia's Lovers — Complete by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4268/4268-0.txt",   #Cousin Phillis by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2522/2522-0.txt",   #A Dark Night's Work by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/7371/7371-0.txt",   #A Sicilian Romance by Ann Ward Radcliffe
    "https://www.gutenberg.org/files/64701/64701-0.txt", #The Romance of the Forest, interspersed with some pieces of poetry. by Radcliffe
    "https://www.gutenberg.org/files/66545/66545-0.txt", #The Heir of Mondolfo by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/31180/31180-0.txt", #Ellen Middleton—A Tale by Georgiana Fullerton
    "https://www.gutenberg.org/files/3322/3322-0.txt",   #East Lynne by Mrs. Henry Wood
    "https://www.gutenberg.org/files/15627/15627-0.txt", #Verner's Pride by Mrs. Henry Wood
    "https://www.gutenberg.org/files/34587/34587-0.txt", #Mrs. Halliburton's Troubles by Mrs. Henry Wood
    "https://www.gutenberg.org/files/22121/22121-0.txt", #Olive: A Novel by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/45975/45975-0.txt", #The Little Lame Prince and His Travelling Cloak by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/21767/21767-0.txt", #Agatha's Husband: A Novel by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/2351/2351-0.txt",   #John Halifax, Gentleman by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/13461/13461-0.txt", #Mistress and Maid: A Household Story by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/14708/14708-0.txt", #The Laurel Bush: An Old-Fashioned Love Story by Dinah Maria Mulock Craik
    {'url': 'https://www.gutenberg.org/files/36157/36157-0.txt', 'work_id': 'daisy burns by julia kavanagh'}, #Daisy Burns (Volume 1) by Julia Kavanagh
    {'url': 'https://www.gutenberg.org/files/36158/36158-0.txt', 'work_id': 'daisy burns by julia kavanagh'}, #Daisy Burns (Volume 2) by Julia Kavanagh
    "https://www.gutenberg.org/files/36160/36160-0.txt", #Rachel Gray: A Tale Founded on Fact by Julia Kavanagh
    "https://www.gutenberg.org/files/1952/1952-0.txt",   #The Yellow Wallpaper by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/3016/3016-0.txt",   #What Diantha Did by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/6053/6053-0.txt",   #Evelina, Or, the History of a Young Lady's Entrance into the World by Fanny Burney
    "https://www.gutenberg.org/files/40619/40619-0.txt", #Camilla; or, A Picture of Youth by Fanny Burney
    "https://www.gutenberg.org/files/171/171-0.txt",     #Charlotte Temple by Mrs. Rowson
    "https://www.gutenberg.org/files/5182/5182-0.txt",   #The Old English Baron: a Gothic Story by Clara Reeve
    "https://www.gutenberg.org/files/23810/23810-0.txt", #At Fault by Kate Chopin
    {'url': 'https://www.gutenberg.org/files/6346/6346-0.txt', 'work_id': 'cecilia or memoirs of an heiress by fanny burney'},   #Cecilia; Or, Memoirs of an Heiress — Volume 1 by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/7146/7146-0.txt', 'work_id': 'cecilia or memoirs of an heiress by fanny burney'},   #Cecilia; Or, Memoirs of an Heiress — Volume 2 by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/7152/7152-0.txt', 'work_id': 'cecilia or memoirs of an heiress by fanny burney'},   #Cecilia; Or, Memoirs of an Heiress — Volume 3 by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/37437/37437-0.txt', 'work_id': 'the wanderer or female difficulties by fanny burney'}, #The Wanderer; or, Female Difficulties (Volume 1 of 5) by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/37438/37438-0.txt', 'work_id': 'the wanderer or female difficulties by fanny burney'}, #The Wanderer; or, Female Difficulties (Volume 2 of 5) by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/37439/37439-0.txt', 'work_id': 'the wanderer or female difficulties by fanny burney'}, #The Wanderer; or, Female Difficulties (Volume 3 of 5) by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/37440/37440-0.txt', 'work_id': 'the wanderer or female difficulties by fanny burney'}, #The Wanderer; or, Female Difficulties (Volume 4 of 5) by Fanny Burney
    {'url': 'https://www.gutenberg.org/files/37441/37441-0.txt', 'work_id': 'the wanderer or female difficulties by fanny burney'}, #The Wanderer; or, Female Difficulties (Volume 5 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/40882/40882-0.txt", #Felix Holt, the Radical by George Eliot
    "https://www.gutenberg.org/files/2165/2165-0.txt",   #The Lifted Veil by George Eliot
    "https://www.gutenberg.org/files/2171/2171-0.txt",   #Brother Jacob by George Eliot
    "https://www.gutenberg.org/files/10762/10762-0.txt", #Impressions of Theophrastus Such by George Eliot
    "https://www.gutenberg.org/files/2547/2547-0.txt",   #Half a Life-Time Ago by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/11371/11371-0.txt", #The Moorland Cottage by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2549/2549-0.txt",   #The Doom of the Griffiths by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/50054/50054-0.txt", #The Female Quixote; or, The Adventures of Arabella, v. 1-2 by Charlotte Lennox
    "https://www.gutenberg.org/files/45187/45187-0.txt", #The Household of Sir Thomas More by Anne Manning
    "https://www.gutenberg.org/files/61080/61080-0.txt", #Cherry & Violet: A Tale of the Great Plague by Anne Manning
    "https://www.gutenberg.org/files/51829/51829-0.txt", #The Old Chelsea Bun-House: A Tale of the Last Century by Anne Manning
    "https://www.gutenberg.org/files/49290/49290-0.txt", #The Year Nine: A Tale of the Tyrol by Anne Manning
    "https://www.gutenberg.org/files/42296/42296-0.txt", #The Duchess of Trajetto by Anne Manning
    "https://www.gutenberg.org/files/67612/67612-0.txt", #Love in Excess; or, the Fatal Enquiry by Eliza Fowler Haywood
    "https://www.gutenberg.org/files/46346/46346-0.txt", #The History of Miss Betsy Thoughtless by Eliza Fowler Haywood
    "https://www.gutenberg.org/files/10804/10804-0.txt", #The Fortunate Foundlings by Eliza Fowler Haywood
    "https://www.gutenberg.org/files/15455/15455-0.txt", #Life's Progress Through the Passions; Or, The Adventures of Natura by Haywood
    "https://www.gutenberg.org/files/1424/1424-0.txt",   #Castle Rackrent by Maria Edgeworth
    "https://www.gutenberg.org/files/1473/1473-0.txt",   #The Absentee by Maria Edgeworth
    "https://www.gutenberg.org/files/271/271-0.txt",     #Black Beauty : The autobiography of a horse by Anna Sewell
    "https://www.gutenberg.org/files/479/479-0.txt",     #Little Lord Fauntleroy by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/25810/25810-0.txt", #In Connection with the De Willoughby Claim by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/137/137-0.txt",     #Sara Crewe; Or, What Happened at Miss Minchin's Boarding School by Burnett
    "https://www.gutenberg.org/files/35300/35300-0.txt", #Louisiana by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/15651/15651-0.txt", #His Grace of Osmonde by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/1550/1550-0.txt",   #A Lady of Quality by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/9487/9487-0.txt",   #A Fair Barbarian by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/25726/25726-0.txt", #The pretty sister of José by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/25725/25725-0.txt", #That Lass O' Lowrie's by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/25727/25727-0.txt", #Vagabondia by Frances Hodgson Burnett
    "https://www.gutenberg.org/files/1212/1212-0.txt",   #Love and Freindship [sic] by Jane Austen
    "https://www.gutenberg.org/files/2521/2521-0.txt",   #Lizzie Leigh by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2548/2548-0.txt",   #The Poor Clare by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/17780/17780-0.txt", #Scenes of Clerical Life by George Eliot
    "https://www.gutenberg.org/files/30273/30273-0.txt", #Tom and Maggie Tulliver by George Eliot
    {'url': 'https://www.gutenberg.org/files/74643/74643-0.txt', 'work_id': 'the italian or the confessional of the black penitents'}, #The Italian, Volume 1 (of 3) : or, the confessional of the Black Penitents
    {'url': 'https://www.gutenberg.org/files/74640/74640-0.txt', 'work_id': 'the italian or the confessional of the black penitents'}, #The Italian, Volume 2 (of 3) : or, the confessional of the Black Penitents
    "https://www.gutenberg.org/files/66749/66749-0.txt", #The Fortunes of Perkin Warbeck: a romance by Mary Wollstonecraft Shelley
    {'url': 'https://www.gutenberg.org/files/63337/63337-0.txt', 'work_id': 'valperga by mary wollstonecraft shelley'}, #Valperga Volume 1 (of 3) by Mary Wollstonecraft Shelley
    {'url': 'https://www.gutenberg.org/files/63338/63338-0.txt', 'work_id': 'valperga by mary wollstonecraft shelley'}, #Valperga Volume 2 (of 3) by Mary Wollstonecraft Shelley
    {'url': 'https://www.gutenberg.org/files/63339/63339-0.txt', 'work_id': 'valperga by mary wollstonecraft shelley'}, #Valperga Volume 3 (of 3) by Mary Wollstonecraft Shelley
    {'url': 'https://www.gutenberg.org/files/64555/64555-0.txt', 'work_id': 'lodore by mary wollstonecraft shelley'}, #Lodore, Vol. 1 (of 3) by Mary Wollstonecraft Shelley
    {'url': 'https://www.gutenberg.org/files/64556/64556-0.txt', 'work_id': 'lodore by mary wollstonecraft shelley'}, #Lodore, Vol. 2 (of 3) by Mary Wollstonecraft Shelley
    {'url': 'https://www.gutenberg.org/files/64557/64557-0.txt', 'work_id': 'lodore by mary wollstonecraft shelley'}, #Lodore, Vol. 3 (of 3) by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/35638/35638-0.txt", #Leonora by Maria Edgeworth
    "https://www.gutenberg.org/files/2129/2129-0.txt",   #Murad the Unlucky, and Other Tales by Maria Edgeworth
    "https://www.gutenberg.org/files/62219/62219-0.txt", #The Day of Small Things by Anne Manning
    "https://www.gutenberg.org/files/51557/51557-0.txt", #The Colloquies of Edward Osborne, Citizen and Clothworker of London by Anne Manning
    "https://www.gutenberg.org/files/35548/35548-0.txt", #Doctor Cupid: A Novel by Rhoda Broughton
    "https://www.gutenberg.org/files/12304/12304-0.txt", #Nancy: A Novel by Rhoda Broughton
    "https://www.gutenberg.org/files/45178/45178-0.txt", #Red as a Rose is She: A Novel by Rhoda Broughton
    "https://www.gutenberg.org/files/34428/34428-0.txt", #Alas! A Novel by Rhoda Broughton
    "https://www.gutenberg.org/files/18459/18459-0.txt", #Dainty's Cruel Rivals; Or, The Fatal Birthday by Mrs. Alex. McVeigh Miller
    "https://www.gutenberg.org/files/7184/7184-0.txt",   #A Heart-Song of To-day (Disturbed by Fire from the 'Unruly Member'): A Novel
    "https://www.gutenberg.org/files/23789/23789-0.txt", #Cruel as the grave by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/42283/42283-0.txt", #The San Rosario Ranch by Maud Howe Elliott
    "https://www.gutenberg.org/files/3823/3823-0.txt",   #Thelma by Marie Corelli
    "https://www.gutenberg.org/files/42332/42332-0.txt", #The Sorrows of Satan by Marie Corelli
    "https://www.gutenberg.org/files/4360/4360-0.txt",   #Vendetta: A Story of One Forgotten by Marie Corelli
    "https://www.gutenberg.org/files/4394/4394-0.txt",   #A Romance of Two Worlds: A Novel by Marie Corelli
    "https://www.gutenberg.org/files/53097/53097-0.txt", #The Murder of Delicia by Marie Corelli
    "https://www.gutenberg.org/files/5114/5114-0.txt",   #Ardath: The Story of a Dead Self by Marie Corelli
    "https://www.gutenberg.org/files/68771/68771-0.txt", #The soul of Lilith by Marie Corelli
    "https://www.gutenberg.org/files/5079/5079-0.txt",   #Ziska: The Problem of a Wicked Soul by Marie Corelli
    "https://www.gutenberg.org/files/49806/49806-0.txt", #Addie's Husband; or, Through clouds to sunshine by Mrs. Gordon Smythies
    "https://www.gutenberg.org/files/36360/36360-0.txt", #Faithful Margaret: A Novel by Mrs. J. M. Simpson
    "https://www.gutenberg.org/files/76634/76634-0.txt", #Lillian's vow : or, The mystery of Raleigh House by Mrs. E. Burke Collins
    "https://www.gutenberg.org/files/76981/76981-0.txt", #Saved from herself : or, On the edge of doom by Adelaide Stirling
    "https://www.gutenberg.org/files/26259/26259-0.txt", #Her mother's secret by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/15774/15774-0.txt", #Ishmael; Or, In the Depths by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/29866/29866-0.txt", #Hidden hand by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/76591/76591-0.txt", #The deserted wife by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/29729/29729-0.txt", #Victor's Triumph by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/16094/16094-0.txt", #For Woman's Love by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/69828/69828-0.txt", #The bride's fate by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/3792/3792-0.txt",   #Capitola the madcap by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/16039/16039-0.txt", #The Lost Lady of Lone by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/75497/75497-0.txt", #The Lady of the Isle : or, the Island Princess by Southworth
    "https://www.gutenberg.org/files/69714/69714-0.txt", #The discarded daughter; or, The children of the isle by Southworth
    "https://www.gutenberg.org/files/70811/70811-0.txt", #Gloria : A novel by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/36713/36713-0.txt", #The Haunted Homestead: A Novel by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/69809/69809-0.txt", #For whose sake? by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/68610/68610-0.txt", #Love's labor won by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/70131/70131-0.txt", #How he won her by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/24337/24337-0.txt", #Capitola's peril by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/71098/71098-0.txt", #Dorothy Harcourt's secret : Sequel to "A deed without a name" by Southworth
    "https://www.gutenberg.org/files/70476/70476-0.txt", #India : the pearl of Pearl River by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/70855/70855-0.txt", #Lilith : A novel by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/70776/70776-0.txt", #The curse of Clifton : or, the widowed bride by Southworth
    "https://www.gutenberg.org/files/68273/68273-0.txt", #Love's bitterest cup by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/32757/32757-0.txt", #Tried for Her Life by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/14382/14382-0.txt", #The Missing Bride by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/69675/69675-0.txt", #Allworth Abbey by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/75006/75006-0.txt", #Fair play : a novel by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/69972/69972-0.txt", #The changed brides by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/6376/6376-0.txt",   #Self-Raised; Or, From the Depths by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/75317/75317-0.txt", #When shadows die by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/76916/76916-0.txt", #Em's husband : A sequel to "Em" by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/77295/77295-0.txt", #Brandon Coyle's wife : a sequel to "A skeleton in the closet" by Southworth
    "https://www.gutenberg.org/files/79290/79290-0.txt", #The fatal marriage : or, Orville Deville by Emma Dorothy Eliza Nevitte Southworth
    "https://www.gutenberg.org/files/22047/22047-0.txt", #The Love Affairs of an Old Maid by Lilian Bell
    "https://www.gutenberg.org/files/41646/41646-0.txt", #Emmeline, the Orphan of the Castle by Charlotte Smith
    "https://www.gutenberg.org/files/2601/2601-0.txt",   #Heartsease; Or, The Brother's Wife by Charlotte M. Yonge
    "https://www.gutenberg.org/files/5700/5700-0.txt",   #Love and Life: An Old Story in Eighteenth Century Costume by Charlotte M. Yonge
    "https://www.gutenberg.org/files/6006/6006-0.txt",   #Under the Storm by Charlotte M. Yonge
    "https://www.gutenberg.org/files/3610/3610-0.txt",   #The Daisy Chain, or Aspirations by Charlotte M. Yonge
    "https://www.gutenberg.org/files/2505/2505-0.txt",   #The Heir of Redclyffe by Charlotte M. Yonge
    "https://www.gutenberg.org/files/3292/3292-0.txt",   #The Clever Woman of the Family by Charlotte M. Yonge
    "https://www.gutenberg.org/files/3048/3048-0.txt",   #The Little Duke: Richard the Fearless by Charlotte M. Yonge
    "https://www.gutenberg.org/files/5080/5080-0.txt",   #Magnum Bonum; Or, Mother Carey's Brood by Charlotte M. Yonge
    "https://www.gutenberg.org/files/26156/26156-0.txt", #Hopes and Fears by Charlotte M. Yonge
    "https://www.gutenberg.org/files/203/203-0.txt",     #Uncle Tom's Cabin by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/12354/12354-0.txt", #Pink and White Tyranny by Harriet Beecher Stowe
]

CLEANING_RULES = {}  # Example: {"123": {"raw_sha256": "...", "start": {"text": "CHAPTER I."}}}

# %% Cell 3: Downloading and cleaning functions
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

class CorpusError(Exception):
    """Invalid corpus input, download, or unresolved quality issue."""


@dataclass(frozen=True)
class Book:
    id: str
    url: str
    note: str = ""
    work_id: str = ""


def gutenberg_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in {"gutenberg.org", "www.gutenberg.org"} or parsed.query or parsed.fragment:
        raise CorpusError(f"Expected an HTTPS Gutenberg URL: {url!r}")
    patterns = (r"/files/(\d+)/\1(?:-[08])?\.txt", r"/cache/epub/(\d+)/pg\1\.txt", r"/ebooks/(\d+)(?:\.txt\.utf-8)?")
    for pattern in patterns:
        match = re.fullmatch(pattern, parsed.path)
        if match:
            return match[1]
    raise CorpusError(f"Unrecognized Gutenberg URL: {url!r}")


def split_for(work_id: str, seed: str, validation: float, test: float) -> str:
    value = int(hashlib.sha256(f"{seed}\0{work_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value < test else "validation" if value < test + validation else "train"


VERSION = "2.0"
DEFAULT_MIRROR = "https://gutenberg.pglaf.org"
START = re.compile(r"(?im)^[ \t]*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT)\b[^\n]*\*{3}[^\n]*$")
END = re.compile(r"(?im)^[ \t]*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT)\b[^\n]*\*{3}[^\n]*$")
LEGACY_END = re.compile(r"(?im)^[ \t]*End of (?:the )?Project Gutenberg(?:'s)? (?:EBook|Etext)\b[^\n]*$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def decode_book(data: bytes) -> tuple[str, str]:
    # Never use errors='ignore' or errors='replace': lost characters are training errors.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252"), "cp1252"
        except UnicodeDecodeError as exc:
            raise CorpusError(f"Cannot decode book without losing characters: {exc}") from exc


def normalize_source(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).lstrip("\ufeff")


def gutenberg_body(text: str) -> tuple[str, dict]:
    text = normalize_source(text)
    if re.search(r"(?is)<(?:!doctype\s+html|html|body)\b", text[:2048]):
        raise CorpusError("HTML response, not a plain-text book.")
    starts = list(START.finditer(text))
    ends = list(END.finditer(text)) or list(LEGACY_END.finditer(text))
    if len(starts) != 1 or len(ends) != 1 or ends[0].start() <= starts[0].end():
        raise CorpusError("Missing, ambiguous, or reversed Gutenberg start/end markers; refusing to guess.")
    start, end = starts[0], ends[0]
    header = text[:start.start()]
    metadata = {}
    for key in ("Title", "Author", "Language", "Release date", "Release Date"):
        m = re.search(rf"(?im)^{re.escape(key)}:\s*(.+)$", header)
        if m:
            metadata[key.lower().replace(" ", "_")] = m[1].strip()
    metadata.update(start_marker=start.group().strip(), end_marker=end.group().strip(), wrapper_removed_chars=len(text) - (end.start() - start.end()))
    return text[start.end():end.start()].strip(), metadata


class Downloader:
    def __init__(self, root: Path, mirror: str, delay: float, timeout: float, retries: int, offline: bool, cache_raw: bool = False):
        self.root, self.mirror = root, mirror.rstrip("/")
        self.delay, self.timeout, self.retries, self.offline = delay, timeout, retries, offline
        self.last_request = 0.0
        self.cache_raw = cache_raw
        self.blocked_reason = None

    def candidates(self, book: Book) -> list[str]:
        digits = "/".join(book.id[:-1]) if len(book.id) > 1 else "0"
        base = f"{self.mirror}/{digits}/{book.id}"
        return [f"{base}/{book.id}-0.txt", f"{self.mirror}/cache/epub/{book.id}/pg{book.id}.txt", f"{base}/{book.id}-8.txt", f"{base}/{book.id}.txt"]

    def fetch(self, book: Book) -> tuple[bytes, dict]:
        raw = self.root / "raw" / f"{book.id}.txt"
        meta_path = raw.with_suffix(".json")
        if raw.exists() and meta_path.exists():
            data, meta = raw.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("sha256") != sha256(data):
                raise CorpusError(f"Cached download checksum mismatch: {raw}")
            if meta.get("id") != book.id or meta.get("source_url") != book.url:
                raise CorpusError(f"Cached download identity mismatch: {raw}")
            return data, meta
        if self.offline:
            raise CorpusError(f"No complete cached download for {book.id}; run without --offline.")
        if self.blocked_reason:
            raise CorpusError(self.blocked_reason)
        missing = []
        for url in self.candidates(book):
            for attempt in range(self.retries + 1):
                time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
                self.last_request = time.monotonic()
                request = Request(url, headers={"User-Agent": f"GutenbergCorpusPrep/{VERSION} (sequential; cached; research corpus)", "Accept": "text/plain"})
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        data = response.read(30 * 1024 * 1024 + 1)
                        if len(data) > 30 * 1024 * 1024:
                            raise CorpusError("Response exceeds the 30 MiB per-book limit.")
                        if "html" in response.headers.get("Content-Type", "").lower():
                            raise CorpusError(f"Server returned HTML for {url}")
                        resolved_url = response.url
                    decoded, encoding = decode_book(data)
                    _, wrapper = gutenberg_body(decoded)
                    for marker in (wrapper["start_marker"], wrapper["end_marker"]):
                        mid = re.search(r"(?:EBOOK|ETEXT)\s+(\d+)\s*\*", marker, re.I)
                        if mid and mid[1] != book.id:
                            raise CorpusError(f"Downloaded ID {mid[1]} does not match requested {book.id}.")
                    meta = {"id": book.id, "source_url": book.url, "download_url": resolved_url, "sha256": sha256(data), "bytes": len(data), "encoding": encoding, "downloaded_at": datetime.now(timezone.utc).isoformat()}
                    if self.cache_raw:
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        tmp = raw.with_suffix(".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(raw)
                        write_json(meta_path, meta)
                    return data, meta
                except HTTPError as exc:
                    if exc.code in (404, 410):
                        missing.append(f"{exc.code}: {url}")
                        break
                    if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                        retry_after = exc.headers.get("Retry-After", "")
                        pause = retry_delay(retry_after, attempt)
                        if pause > 300:
                            self.blocked_reason = "Server requested an extended pause; retry this run later."
                            raise CorpusError(f"Server requested a {pause:.0f}s pause; retry this book later.") from exc
                        time.sleep(pause)
                        continue
                    if exc.code in (401, 403, 429):
                        self.blocked_reason = f"Server refused access (HTTP {exc.code}); further network requests stopped."
                    raise CorpusError(f"HTTP {exc.code}: {url}") from exc
                except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
                    if attempt < self.retries:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise CorpusError(f"Download failed: {url}: {exc}") from exc
        raise CorpusError("No plain-text edition found. " + "; ".join(missing))


def retry_delay(value: str, attempt: int) -> float:
    if value.isdigit():
        return float(value)
    if value:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            pass
    return 2 ** (attempt + 2)


CHAPTER = re.compile(
    r"(?im)^[ \t]*(?:CHAPTER|CHAP\.)[ \t]+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FIRST|SECOND|THIRD)"
    r"\b[^\n]{0,140}$"
)
PROLOGUE = re.compile(r"(?im)^[ \t]*(?:PROLOGUE|PRELUDE|INTRODUCTORY CHAPTER)[. \t]*$")
SUSPECT = {
    "gutenberg_boilerplate": r"(?i)project\s+gutenberg|gutenberg\.org|www\.pgdp\.net|distributed\s+proofread",
    "transcriber_material": r"(?i)transcrib(?:er|ers|er's|ers’|er’s).{0,15}(?:note|comment)|digitiz(?:ed|ation)|scann?ed\s+(?:by|from)",
    "contents_or_illustration_list": r"(?im)^\s*(?:(?:table of )?contents|list of illustrations|illustrations)[. \t]*$",
    "publisher_material": r"(?im)^\s*(?:all rights reserved|printed (?:by|in)|published by|copyright(?:ed)?\b|advertisements\b)",
    "possible_back_matter": r"(?im)^\s*(?:by the same author|other books by|books by|publisher['’]s (?:note|catalogue|catalog|announcement)|appendix|editor['’]s notes?)[.: \t]*$",
    "remaining_illustration": r"(?i)\[illustration\b|\[page\s+\d+\]|\[pg\.?\s*\d+\]",
    "possible_footnote": r"(?im)\[(?:\d{1,4}|[A-Z]|[*†‡])\]|^\s*(?:footnotes?|endnotes?)\b",
    "markup": r"(?i)</?(?:html|body|div|span|p|a|img)\b[^>]*>|&(?:amp|lt|gt|nbsp);",
    "replacement_character": "\ufffd",
    "possible_encoding_damage": r"Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â[€\u0080-\u009f]",
    "control_character": r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]",
}


def record_removal(audit: list, reason: str, removed: str) -> None:
    if removed:
        audit.append({"reason": reason, "chars": len(removed), "sha256": sha256(removed.encode("utf-8")), "preview": removed[:200]})


def substitute(text: str, pattern: str, reason: str, audit: list, replacement: str = "") -> str:
    removed_hash = hashlib.sha256()
    total_chars, count, preview = 0, 0, ""
    def replace(match):
        nonlocal total_chars, count, preview
        removed = match.group()
        removed_hash.update(removed.encode("utf-8"))
        total_chars += len(removed)
        count += 1
        if len(preview) < 200:
            preview += removed[:200 - len(preview)]
        return replacement
    result = re.sub(pattern, replace, text)
    if count:
        audit.append({"reason": reason, "count": count, "chars": total_chars, "sha256": removed_hash.hexdigest(), "preview": preview})
    return result


def anchor_position(text: str, spec: dict) -> int:
    if not isinstance(spec, dict) or not isinstance(spec.get("text"), str) or not spec["text"]:
        raise CorpusError("An anchor requires a nonempty literal 'text' field.")
    positions = [m.start() for m in re.finditer(re.escape(normalize_source(spec["text"])), text)]
    occurrence = spec.get("occurrence")
    if occurrence is None:
        if len(positions) != 1:
            raise CorpusError(f"Anchor must match exactly once (found {len(positions)}): {spec['text'][:80]!r}")
        return positions[0]
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or not 1 <= occurrence <= len(positions):
        raise CorpusError(f"Invalid occurrence for anchor {spec['text'][:80]!r}")
    return positions[occurrence - 1]


def trim_to_narrative(text: str, rules: dict, audit: list) -> tuple[str, list[str], str]:
    problems = []
    method = "explicit"
    if "start" in rules:
        start = anchor_position(text, rules["start"])
    else:
        # TOC entries are followed by more short entries, not sustained prose.
        headings = list(CHAPTER.finditer(text))
        candidates = []
        for index, heading in enumerate(headings):
            next_pos = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            after = text[heading.end():min(next_pos, heading.end() + 5000)]
            paragraphs = re.split(r"\n[ \t]*\n", after.strip())[:5]
            if any(len(p.split()) >= 45 and re.search(r"[.!?][\"’”']?(?:\s|$)", p) for p in paragraphs):
                candidates.append(heading)
        if candidates:
            start = candidates[0].start()
            method = "chapter_heuristic"
            prologues = list(PROLOGUE.finditer(text[:start]))
            if prologues:
                start = prologues[-1].start()
            if start > len(text) * 0.25:
                problems.append("large_front_matter_cut")
            prefix = text[:start]
            if re.search(r"(?im)^\s*(?:LETTER\s+(?:I|1)|PREFATORY TALE)\b", prefix):
                problems.append("possible_narrative_before_first_chapter")
        else:
            start = 0
            method = "unresolved"
            problems.append("narrative_start_not_identified")
    end = anchor_position(text, rules["end"]) if "end" in rules else len(text)
    if end <= start:
        raise CorpusError("The narrative end anchor precedes its start anchor.")
    record_removal(audit, "front_matter", text[:start])
    record_removal(audit, "back_matter", text[end:])
    return text[start:end], problems, method


def tidy_layout(text: str, audit: list, reflow: bool) -> str:
    # NFC, not NFKC: preserve ligatures, accents, spelling, case, and punctuation.
    text = text.replace("\u00a0", " ").replace("\u00ad", "").replace("\ufeff", "")
    text = text.replace("\f", "\n\n")
    text = substitute(text, r"(?m)^[ \t]*(?:\[Page\s+\d+\]|\[Pg\.?\s*\d+\])[ \t]*$", "page_label", audit)
    # Underscores are Gutenberg's emphasis notation; keep the enclosed words.
    # Strip paired emphasis only; preserve literal underscores in words.
    def unemphasize(match):
        record_removal(audit, "emphasis_marker", "__")
        return match[1]
    text = re.sub(r"(?<!\w)_(?=\S)([^_]+?)(?<=\S)_(?!\w)", unemphasize, text)
    paragraphs = []
    for paragraph in re.split(r"\n[ \t]*\n+", text.strip()):
        lines = [line.rstrip() for line in paragraph.splitlines()]
        if not lines:
            continue
        # Reflow wrapped prose only. Short/indented lines may be verse or a letter.
        prose = len(lines) > 1 and all(len(line.strip()) >= 45 for line in lines[:-1])
        indented = any(line.startswith(("  ", "\t")) for line in lines[1:])
        if reflow and prose and not indented:
            joined = "\n".join(line.strip() for line in lines)
            # Keep printed hyphens, but avoid inserting a space into well-\nknown.
            joined = re.sub(r"(?<=\w)-\n(?=\w)", "-", joined)
            paragraphs.append(joined.replace("\n", " "))
        else:
            paragraphs.append("\n".join(lines).strip())
    return "\n\n".join(p for p in paragraphs if p).strip() + "\n"


def clean_book(data: bytes, rules: dict, keep_book_matter: bool = False, reflow: bool = True) -> tuple[str, dict]:
    decoded, encoding = decode_book(data)
    body, metadata = gutenberg_body(decoded)
    audit, problems = [], []
    if rules and rules.get("raw_sha256") != sha256(data):
        raise CorpusError("Cleaning rules require the matching raw_sha256; the edition may have changed.")
    original_chars = len(body)
    # Explicit digital artifacts can occur within either boundary.
    body = substitute(body, r"(?is)\[(?:Illustration\b|Illustrations\b)[^\[\]]{0,12000}\]", "illustration_caption", audit)
    body = substitute(body, r"(?is)\[Transcriber[’']?s?\s+Notes?\b[^\[\]]{0,12000}\]", "bracketed_transcriber_note", audit)
    # A terminal transcriber's section is safely bounded only near the end.
    notes = list(re.finditer(r"(?im)^[ \t]*(?:[*_ ]*)TRANSCRIBER[’']?S?[ \t]+NOTES?[.: \t*_]*$", body))
    if notes and notes[-1].start() > len(body) * 0.8:
        record_removal(audit, "terminal_transcriber_notes", body[notes[-1].start():])
        body = body[:notes[-1].start()]
    # Apply explicit anchors to the normalized, artifact-stripped text.
    if not keep_book_matter:
        body, boundary_problems, method = trim_to_narrative(body, rules, audit)
        problems.extend(boundary_problems)
    else:
        method = "keep_book_matter"
    for removal in rules.get("remove", []):
        if not isinstance(removal, dict) or not isinstance(removal.get("text"), str) or not removal["text"]:
            raise CorpusError("Each removal requires a nonempty literal 'text' field.")
        literal = normalize_source(removal["text"])
        expected = removal.get("count", 1)
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0 or body.count(literal) != expected:
            raise CorpusError(f"Removal count mismatch: {literal[:80]!r}")
        record_removal(audit, "explicit_removal", literal * expected)
        body = body.replace(literal, "")
    if not keep_book_matter:
        # Remove only explicitly labelled footnotes, including their own reference.
        # Bare [1]/[A] references are ambiguous and sent for review, never guessed.
        body = substitute(body, r"(?is)\[Footnote\b[^\[\]]{0,12000}\]", "labelled_footnote", audit)
    clean = tidy_layout(body, audit, reflow)
    findings = {}
    for name, pattern in SUSPECT.items():
        if keep_book_matter and name in {"contents_or_illustration_list", "publisher_material", "possible_footnote", "possible_back_matter"}:
            continue
        hits = list(re.finditer(pattern, clean))
        if hits:
            findings[name] = {"count": len(hits), "examples": [clean[max(0, m.start()-60):m.end()+100] for m in hits[:3]]}
            problems.append(name)
    if encoding != "utf-8":
        problems.append("legacy_encoding_needs_review")
    if len(clean.split()) < 100:
        problems.append("very_short_book")
    if not keep_book_matter and len(clean) < original_chars * 0.5:
        problems.append("more_than_half_removed")
    accepted = rules.get("accept_findings", [])
    if not isinstance(accepted, list) or not all(isinstance(x, str) for x in accepted):
        raise CorpusError("accept_findings must be a list of finding names.")
    unwaivable = {"gutenberg_boilerplate", "replacement_character", "control_character", "markup"}
    problems = sorted(set(problems) - (set(accepted) - unwaivable))
    return clean, {"encoding": encoding, "metadata": metadata, "boundary_method": method, "removals": audit, "findings": findings, "problems": problems, "words": len(clean.split()), "characters": len(clean), "original_body_characters": original_chars}



def load_books(entries):
    books = {}
    for entry in entries:
        fields = {"url": entry} if isinstance(entry, str) else entry
        if not isinstance(fields, dict) or set(fields) - {"url", "note", "work_id"}:
            raise CorpusError("Use a URL or a dictionary with url, note, and work_id.")
        if "url" not in fields or any(not isinstance(v, str) or not v.strip() for v in fields.values()):
            raise CorpusError("Corpus fields must be nonempty strings.")
        book = Book(gutenberg_id(fields["url"]), **fields)
        if book.id in books and books[book.id].work_id != book.work_id:
            raise CorpusError(f"Conflicting work_id for {book.id}.")
        books.setdefault(book.id, book)
    if not books:
        raise CorpusError("The corpus list is empty.")
    return list(books.values())


def prepare_books(entries, rules, output, allow_partial=False, offline=False):
    downloader = Downloader(output, DEFAULT_MIRROR, 2, 30, 2, offline, cache_raw=False)
    documents, audit, seen = [], [], set()
    for book in load_books(entries):
        row = {"id": book.id, "url": book.url}
        try:
            raw, metadata = downloader.fetch(book)
            text, report = clean_book(raw, rules.get(book.id, {}))
            digest = sha256(text.encode())
            row.update(report, raw_sha256=metadata["sha256"])
            row["status"] = "needs_review" if report["problems"] else "duplicate" if digest in seen else "accepted"
            if row["status"] == "accepted":
                seen.add(digest)
                group = rules.get(book.id, {}).get("work_id") or book.work_id or book.id
                documents.append({"id": book.id, "text": text,
                                  "split": split_for(group, "gutenberg-corpus-v1", .05, .05)})
        except (CorpusError, OSError, ValueError) as exc:
            row.update(status="failed", error=str(exc))
        audit.append(row)
        print(book.id, row["status"], row.get("problems", row.get("error", "")))
    write_json(output / "audit.json", audit)
    print(dict(Counter(row["status"] for row in audit)))
    if not allow_partial and any(row["status"] in {"needs_review", "failed"} for row in audit):
        raise CorpusError("Review audit.json, edit CLEANING_RULES and rerun Cell 4; or enable ALLOW_PARTIAL_CORPUS.")
    return documents


# %% Cell 4: Prepare books, train the tokenizer, and create token windows
# Rules are keyed by ebook ID, with raw_sha256 and optional literal start/end/remove.
# Books are downloaded into memory, never into Drive or a corpus directory.
documents = prepare_books(sources, CLEANING_RULES, OUTPUT_DIR, ALLOW_PARTIAL_CORPUS)
books_by_split = {split: [doc["text"] for doc in documents if doc["split"] == split]
                  for split in ("train", "validation", "test")}
if not books_by_split["train"] or not books_by_split["validation"]:
    raise CorpusError("Need training and validation books. Add works to the list; never split a book across sets.")

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]
if any(token in doc["text"] for doc in documents for token in SPECIAL_TOKENS):
    raise CorpusError("A book contains a reserved tokenizer control token.")
if RESUME:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(OUTPUT_DIR / "tokenizer", local_files_only=True)
else:
    if (OUTPUT_DIR / "run.json").exists():
        raise ValueError("This output folder already has a run. Set RESUME=True or choose a new OUTPUT_DIR.")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(books_by_split["train"], vocab_size=config.vocab_size,
                           min_frequency=config.min_frequency, special_tokens=SPECIAL_TOKENS)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer,
                bos_token=SPECIAL_TOKENS[0], eos_token=SPECIAL_TOKENS[0],
                pad_token=SPECIAL_TOKENS[1], model_max_length=10**12)
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")


class TokenWindows:
    """One-token overlap keeps every prediction target, including each book's EOS."""
    def __init__(self, texts, tokenizer, context_length):
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        self.context_length = context_length
        self.books = [np.asarray([tokenizer.bos_token_id,
                      *tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id],
                      dtype=np.int32) for text in texts]
        self.ends = np.cumsum([math.ceil((len(book) - 1) / (context_length - 1)) for book in self.books]).tolist()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        book_index = bisect_right(self.ends, index)
        local = index - (self.ends[book_index - 1] if book_index else 0)
        start = local * (self.context_length - 1)
        return self.books[book_index][start:start + self.context_length].tolist()


def collate_windows(windows, pad_token_id):
    ids = torch.full((len(windows), max(map(len, windows))), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(windows):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


datasets = {split: TokenWindows(texts, tokenizer, config.context_length)
            for split, texts in books_by_split.items()}
print("Books:", {split: len(texts) for split, texts in books_by_split.items()})
print("Token windows:", {split: len(data) for split, data in datasets.items()})

# %% Cell 5: Model, sampling, evaluation, and checkpoint functions
def make_model(config: TrainConfig, tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel
    architecture = GPT2Config(
        vocab_size=len(tokenizer), n_positions=config.context_length,
        n_ctx=config.context_length, n_layer=config.n_layer,
        n_head=config.n_head, n_embd=config.n_embd,
        resid_pdrop=config.dropout, embd_pdrop=config.dropout,
        attn_pdrop=config.dropout, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=False, attn_implementation="sdpa", loss_type="ForCausalLM",
    )
    # This constructor initializes random weights. No pretrained model is loaded.
    return GPT2LMHeadModel(architecture)


def amp_context(device):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def evaluate(model, dataset, config, tokenizer, device):
    import torch
    if not len(dataset):
        return None
    model.eval()
    total_nll, targets = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(dataset), config.batch_size):
            batch = collate_windows([dataset[i] for i in range(start, min(start + config.batch_size, len(dataset)))], tokenizer.pad_token_id)
            count = int((batch["labels"][:, 1:] != -100).sum())
            batch = {key: value.to(device) for key, value in batch.items()}
            with amp_context(device):
                output = model(**batch)
            loss = float(output.loss)
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss.")
            total_nll += loss * count
            targets += count
    loss = total_nll / targets
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 700 else None, "target_tokens": targets}


def generate_samples(model, tokenizer, config: TrainConfig, device, step: int, output: Path):
    """Print/save comparable prose continuations without perturbing training RNG.

    Generation uses the current weights, evaluation mode, and no gradients. The
    caller's model mode and CPU/CUDA random states are restored even on failure.
    """
    import torch

    was_training = model.training
    samples = []
    new_tokens = min(config.sample_max_new_tokens, model.config.n_positions - 1)
    prompt_limit = model.config.n_positions - new_tokens
    # manual_seed seeds all CUDA generators as well as CPU, so fork all of them.
    cuda_devices = list(range(torch.cuda.device_count()))
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            for index, prompt in enumerate(config.sample_prompts):
                torch.manual_seed(config.sample_seed + index)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                truncated = len(prompt_ids) > prompt_limit
                prompt_ids = prompt_ids[-prompt_limit:] or [tokenizer.bos_token_id]
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with amp_context(device):
                    generated = model.generate(
                        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=new_tokens, do_sample=True,
                        temperature=config.sample_temperature, top_p=config.sample_top_p,
                        top_k=0, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bad_words_ids=[[tokenizer.pad_token_id]],
                    )
                continuation_ids = generated[0, len(prompt_ids):].tolist()
                samples.append({
                    "prompt": prompt, "input_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                    "prompt_truncated": truncated,
                    "continuation": tokenizer.decode(continuation_ids, skip_special_tokens=True),
                    "generated_tokens": len(continuation_ids), "seed": config.sample_seed + index,
                })
    finally:
        model.train(was_training)
    record = {"step": step, "max_new_tokens": new_tokens,
              "temperature": config.sample_temperature, "top_p": config.sample_top_p,
              "top_k": 0, "samples": samples}
    with (output / "samples.jsonl").open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"\nSamples at optimizer step {step}", flush=True)
    for index, sample in enumerate(samples, 1):
        notice = " (prompt shortened to fit context)" if sample["prompt_truncated"] else ""
        print(f"\nPrompt {index}{notice}: {sample['input_text']}\nContinuation: {sample['continuation']}", flush=True)
    return record


def checkpoint_paths(output):
    paths = [path for path in output.glob("checkpoint-*") if path.is_dir() and (path / "complete.json").is_file() and path.name.removeprefix("checkpoint-").isdigit()]
    return sorted(paths, key=lambda p: int(p.name.split("-")[-1]))


def save_checkpoint(output, model, optimizer, scheduler, state, identity, config):
    import numpy as np
    import torch
    target = output / f"checkpoint-{state['step']}"
    if target.exists():
        raise CorpusError(f"Checkpoint already exists: {target}")
    staging = output / f".checkpoint-{state['step']}.incomplete"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "state": state, "identity": identity, "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, staging / "training_state.pt")
    write_json(staging / "complete.json", {"step": state["step"], "identity": identity})
    staging.rename(target)
    # Delete only checkpoints generated by this run, after the new one is complete.
    paths = checkpoint_paths(output)
    keep = set(paths[-config.keep_checkpoints:])
    if state["best_step"] is not None:
        keep.add(output / f"checkpoint-{state['best_step']}")
    for path in paths:
        if path not in keep:
            shutil.rmtree(path)
    print(f"Checkpoint saved: {target}", flush=True)


def restore_checkpoint(path, model, optimizer, scheduler, identity, device):
    import numpy as np
    import torch
    from safetensors.torch import load_model
    marker = json.loads((path / "complete.json").read_text())
    if marker["identity"] != identity:
        raise CorpusError("Checkpoint identity does not match this run.")
    load_model(model, str(path / "model.safetensors"), device=str(device))
    # This pickle is created by this script inside the selected run directory.
    # Never resume an untrusted third-party training_state.pt file.
    saved = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if saved["identity"] != identity:
        raise CorpusError("Training state identity mismatch.")
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda" and saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["state"]


def log_metrics(output, values):
    print(json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as target:
        target.write(json.dumps(values, allow_nan=False) + "\n")


# %% Cell 6: Train with the custom PyTorch loop

def train(datasets, tokenizer, config, output, resume=False, max_seconds=7200, device_name="cuda"):
    config.validate()
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds must be nonnegative and finite")
    if not len(datasets["train"]) or not len(datasets["validation"]):
        raise CorpusError("Training and validation sets must both be nonempty.")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise CorpusError("Select an A100 GPU runtime in Colab.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = "CPU (smoke tests only)"
        torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Small in-memory fingerprints replace the old source files, manifests and token cache.
    settings = {k: v for k, v in asdict(config).items() if not k.startswith("sample_")}
    data_hash = hashlib.sha256()
    for split, dataset in datasets.items():
        data_hash.update(split.encode())
        for book in dataset.books:
            data_hash.update(len(book).to_bytes(8, "little"))
            data_hash.update(book.astype("<i4").tobytes())
    recipe = {"config": settings, "data": data_hash.hexdigest(),
              "tokenizer": sha256(tokenizer.backend_tokenizer.to_str().encode()),
              "precision": "bf16" if device.type == "cuda" else "fp32",
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "numpy")}}
    identity = sha256(json.dumps(recipe, sort_keys=True).encode())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        if json.loads((output / "run.json").read_text()) != recipe:
            raise CorpusError("Corpus or settings changed. Choose a new OUTPUT_DIR and set RESUME=False.")
    else:
        if (output / "run.json").exists() or checkpoint_paths(output):
            raise CorpusError("Output already contains a run. Set RESUME=True or choose a new OUTPUT_DIR.")
        write_json(output / "run.json", recipe)
    tokenizer.save_pretrained(output / "tokenizer")
    batches_per_epoch = math.ceil(len(datasets["train"]) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps:
        total_steps = min(total_steps, config.max_steps)
    warmup_steps = min(max(1, round(total_steps * config.warmup_fraction)), total_steps - 1) if config.warmup_fraction else 0

    model = make_model(config, tokenizer).to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")

    def lr_scale(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    state = {"epoch": 0, "next_batch": 0, "step": 0, "tokens_seen": 0, "best_loss": None, "best_step": None}
    checkpoints = checkpoint_paths(output)
    if resume and checkpoints:
        state = restore_checkpoint(checkpoints[-1], model, optimizer, scheduler, identity, device)
        print(f"Resuming optimizer step {state['step']}.", flush=True)
    elif resume:
        print("No complete checkpoint yet; restarting this initialized run from its seed.", flush=True)
        # Tokenizer training may use its own RNG; model initialization uses the seed above.
    summary = {"parameters": sum(p.numel() for p in model.parameters()), "vocab_size": len(tokenizer),
               "splits": {s: {"books": len(datasets[s].books), "target_tokens": sum(len(book) - 1 for book in datasets[s].books), "windows": len(datasets[s])} for s in datasets},
               "planned_optimizer_steps": total_steps, "steps_per_epoch": steps_per_epoch,
               "maximum_tokens_per_update": config.batch_size * config.gradient_accumulation * (config.context_length - 1), "hardware": hardware}
    write_json(output / "training_plan.json", summary)
    print(json.dumps(summary, indent=2), flush=True)

    started, session_tokens = time.monotonic(), 0
    stop_reason = "complete"
    while state["step"] < total_steps and state["epoch"] < config.epochs:
        epoch = state["epoch"]
        order = np.random.default_rng(config.seed + epoch).permutation(len(datasets["train"]))
        for batch_number in range(state["next_batch"], batches_per_epoch, config.gradient_accumulation):
            batch_end = min(batch_number + config.gradient_accumulation, batches_per_epoch)
            microbatches = []
            for number in range(batch_number, batch_end):
                indices = order[number * config.batch_size:(number + 1) * config.batch_size]
                microbatches.append(collate_windows([datasets["train"][int(i)] for i in indices], tokenizer.pad_token_id))
            targets = [int((batch["labels"][:, 1:] != -100).sum()) for batch in microbatches]
            total_targets = sum(targets)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            nll = 0.0
            for batch, count in zip(microbatches, targets):
                batch = {key: value.to(device) for key, value in batch.items()}
                with amp_context(device):
                    result = model(**batch)
                    # Weight by real targets, including a partial final accumulation.
                    loss = result.loss * (count / total_targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; last complete checkpoint remains available.")
                nll += float(result.loss.detach()) * count
                loss.backward()
                del result, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            state["step"] += 1
            state["tokens_seen"] += total_targets
            session_tokens += total_targets
            state["next_batch"] = batch_end
            if batch_end == batches_per_epoch:
                state["epoch"], state["next_batch"] = epoch + 1, 0
            if config.sample_every and state["step"] % config.sample_every == 0:
                generate_samples(model, tokenizer, config, device, state["step"], output)
            elapsed = time.monotonic() - started
            if state["step"] == 1 or state["step"] % config.log_every == 0:
                log_metrics(output, {"step": state["step"], "train_loss": nll / total_targets, "learning_rate": scheduler.get_last_lr()[0], "tokens_seen": state["tokens_seen"], "session_seconds": elapsed, "session_tokens_per_second": session_tokens / max(elapsed, 1e-9)})
            finished = state["step"] >= total_steps or state["epoch"] >= config.epochs
            timed_out = max_seconds > 0 and elapsed >= max_seconds
            should_eval = state["step"] % config.eval_every == 0 or batch_end == batches_per_epoch or finished or timed_out
            if should_eval:
                metrics = evaluate(model, datasets["validation"], config, tokenizer, device)
                log_metrics(output, {"step": state["step"], "validation": metrics})
                if state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
                    state["best_loss"], state["best_step"] = metrics["loss"], state["step"]
            if should_eval or state["step"] % config.save_every == 0 or finished or timed_out:
                save_checkpoint(output, model, optimizer, scheduler, state, identity, config)
            if finished or timed_out:
                stop_reason = "complete" if finished else "session_time_limit"
                break
        if stop_reason == "session_time_limit":
            break

    # Export the validation-selected model; test is touched only at schedule completion.
    from safetensors.torch import load_model
    best = output / f"checkpoint-{state['best_step']}"
    load_model(model, str(best / "model.safetensors"), device=str(device))
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.gradient_checkpointing_disable()
    model.save_pretrained(output / "best_model", safe_serialization=True)
    tokenizer.model_max_length = config.context_length
    tokenizer.save_pretrained(output / "best_model")
    complete = state["step"] >= total_steps or state["epoch"] >= config.epochs
    result = {"complete": complete, "stop_reason": stop_reason, "step": state["step"], "best_step": state["best_step"], "best_validation_loss": state["best_loss"], "tokens_seen": state["tokens_seen"], "session_seconds": time.monotonic() - started, "model": str(output / "best_model")}
    if complete:
        model.config.use_cache = False
        result["test"] = evaluate(model, datasets["test"], config, tokenizer, device)
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return model, result

model, result = train(datasets, tokenizer, config, OUTPUT_DIR, RESUME, MAX_SESSION_SECONDS, DEVICE)

# %% Cell 7: Try your own prompt
PROMPT = "The evening was quiet, and"
sample_config = replace(config, sample_prompts=[PROMPT], sample_max_new_tokens=200)
generate_samples(model, tokenizer, sample_config, next(model.parameters()).device,
                 result["step"], OUTPUT_DIR)

# Optional download of the trained model (uncomment):
# from google.colab import files
# files.download(shutil.make_archive("female-authors-model", "zip", OUTPUT_DIR / "best_model"))
