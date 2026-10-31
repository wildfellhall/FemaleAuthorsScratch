"""FemaleAuthorsScratch — ALL-IN-ONE Colab script; paste numbered cells in order.
Select an A100 runtime. The corpus list is defined here; Google Drive is not used.
Start a fresh run after changing the parser or training recipe. Download outputs
you need to retain before Colab deletes the local runtime.
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
    context_length: int = 2048
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    batch_size: int = 4  # 32 windows/update keeps the token batch near 65K at 2K context
    gradient_accumulation: int = 8
    epochs: int = 12  # ceiling; validation can stop training earlier
    max_steps: int = 0  # zero: derive the schedule from epochs and actual data size
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    warmup_fraction: float = 0.03
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    eval_every: int = 500  # also evaluate at every epoch end
    eval_stride: int = 1024
    early_stopping_patience: int = 3  # full epochs without a meaningful improvement
    early_stopping_min_delta: float = 0.002
    save_every: int = 100
    log_every: int = 10
    keep_checkpoints: int = 2  # best checkpoint is also retained
    gradient_checkpointing: bool = False  # faster on an A100; enable if memory is tight
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
        integers = ("vocab_size", "min_frequency", "context_length", "n_layer", "n_head", "n_embd", "batch_size", "gradient_accumulation", "epochs", "eval_every", "eval_stride", "save_every", "log_every", "keep_checkpoints")
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
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1].")
        if type(self.early_stopping_patience) is not int or self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be a nonnegative integer; 0 disables it.")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be nonnegative and finite.")
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
MAX_SESSION_SECONDS = 0              # Optional time limit; 0 uses epochs/early stopping.
DEVICE = "cuda"                      # "cpu" is for tiny development tests only.
config.validate()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Cell 2: Corpus list and cleaning rules
# Synced from CorpusCollection/femalecorpus.py at f6fe26fd (2026-10-30).

sources = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",   #Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/files/3268/3268-0.txt",   #The Mysteries of Udolpho by Ann Ward Radcliffe
    "https://www.gutenberg.org/files/21839/21839-0.txt", #Sense and Sensibility by Jane Austen
    "https://www.gutenberg.org/files/121/121-0.txt",     #Northanger Abbey by Jane Austen
    "https://www.gutenberg.org/files/158/158-0.txt",     #Emma by Jane Austen
    "https://www.gutenberg.org/files/105/105-0.txt",     #Persuasion by Jane Austen
    "https://www.gutenberg.org/files/141/141-0.txt",     #Mansfield Park by Jane Austen
    "https://www.gutenberg.org/files/145/145-0.txt",     #Middlemarch by George Eliot
    "https://www.gutenberg.org/files/550/550-0.txt",     #Silas Marner by George Eliot
    "https://www.gutenberg.org/files/507/507-0.txt",     #Adam Bede by George Eliot
    "https://www.gutenberg.org/files/6688/6688-0.txt",   #The Mill on the Floss by George Eliot
    "https://www.gutenberg.org/files/7469/7469-0.txt",   #Daniel Deronda by George Eliot
    "https://www.gutenberg.org/files/24020/24020-0.txt", #Romola by George Eliot
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
    "https://www.gutenberg.org/files/58188/58188-0.txt", #Oswald Cray: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/38832/38832-0.txt", #A Life's Secret: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/57875/57875-0.txt", #Anne Hereford: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58086/58086-0.txt", #Roland Yorke by Mrs. Henry Wood
    "https://www.gutenberg.org/files/36106/36106-0.txt", #Trevlyn Hold: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/42200/42200-0.txt", #The Shadow of Ashlydyat by Mrs. Henry Wood
    "https://www.gutenberg.org/files/9192/9192-0.txt",   #The Channings: A Story by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58582/58582-0.txt", #St. Martin's Eve: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58431/58431-0.txt", #Orville College: A Story by Mrs. Henry Wood
    "https://www.gutenberg.org/files/59144/59144-0.txt", #The Master of Greylands: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/39692/39692-0.txt", #Mildred Arkell: A Novel. Vol. 1 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/39377/39377-0.txt", #Mildred Arkell: A Novel. Vol. 2 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/39693/39693-0.txt", #Mildred Arkell: A Novel. Vol. 3 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/40418/40418-0.txt", #It May Be True, Vol. 1 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/39192/39192-0.txt", #It May Be True, Vol. 2 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/39193/39193-0.txt", #It May Be True, Vol. 3 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/38623/38623-0.txt", #The Story of Charles Strange: A Novel. Vol. 1 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/38624/38624-0.txt", #The Story of Charles Strange: A Novel. Vol. 2 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/38625/38625-0.txt", #The Story of Charles Strange: A Novel. Vol. 3 (of 3) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58345/58345-0.txt", #Within the Maze: A Novel, Vol. 1 (of 2) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58346/58346-0.txt", #Within the Maze: A Novel, Vol. 2 (of 2) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58046/58046-0.txt", #The Red Court Farm: A Novel (Vol. 1 of 2) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58047/58047-0.txt", #The Red Court Farm: A Novel (Vol. 2 of 2) by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58774/58774-0.txt", #Court Netherleigh: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58288/58288-0.txt", #Bessy Rane: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/58701/58701-0.txt", #Edina: A Novel by Mrs. Henry Wood
    "https://www.gutenberg.org/files/22121/22121-0.txt", #Olive: A Novel by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/45975/45975-0.txt", #The Little Lame Prince and His Travelling Cloak by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/21767/21767-0.txt", #Agatha's Husband: A Novel by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/2351/2351-0.txt",   #John Halifax, Gentleman by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/13461/13461-0.txt", #Mistress and Maid: A Household Story by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/14708/14708-0.txt", #The Laurel Bush: An Old-Fashioned Love Story by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/36157/36157-0.txt", #Daisy Burns (Volume 1) by Julia Kavanagh
    "https://www.gutenberg.org/files/36158/36158-0.txt", #Daisy Burns (Volume 2) by Julia Kavanagh
    "https://www.gutenberg.org/files/36160/36160-0.txt", #Rachel Gray: A Tale Founded on Fact by Julia Kavanagh
    "https://www.gutenberg.org/files/1952/1952-0.txt",   #The Yellow Wallpaper by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/3016/3016-0.txt",   #What Diantha Did by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/6053/6053-0.txt",   #Evelina, Or, the History of a Young Lady's Entrance into the World by Fanny Burney
    "https://www.gutenberg.org/files/40619/40619-0.txt", #Camilla; or, A Picture of Youth by Fanny Burney
    "https://www.gutenberg.org/files/171/171-0.txt",     #Charlotte Temple by Mrs. Rowson
    "https://www.gutenberg.org/files/5182/5182-0.txt",   #The Old English Baron: a Gothic Story by Clara Reeve
    "https://www.gutenberg.org/files/23810/23810-0.txt", #At Fault by Kate Chopin
    "https://www.gutenberg.org/files/6346/6346-0.txt",   #Cecilia; Or, Memoirs of an Heiress — Volume 1 by Fanny Burney
    "https://www.gutenberg.org/files/7146/7146-0.txt",   #Cecilia; Or, Memoirs of an Heiress — Volume 2 by Fanny Burney
    "https://www.gutenberg.org/files/7152/7152-0.txt",   #Cecilia; Or, Memoirs of an Heiress — Volume 3 by Fanny Burney
    "https://www.gutenberg.org/files/37437/37437-0.txt", #The Wanderer; or, Female Difficulties (Volume 1 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37438/37438-0.txt", #The Wanderer; or, Female Difficulties (Volume 2 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37439/37439-0.txt", #The Wanderer; or, Female Difficulties (Volume 3 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37440/37440-0.txt", #The Wanderer; or, Female Difficulties (Volume 4 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37441/37441-0.txt", #The Wanderer; or, Female Difficulties (Volume 5 of 5) by Fanny Burney
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
    "https://www.gutenberg.org/files/74643/74643-0.txt", #The Italian, Volume 1 (of 3) : or, the confessional of the Black Penitents
    "https://www.gutenberg.org/files/74640/74640-0.txt", #The Italian, Volume 2 (of 3) : or, the confessional of the Black Penitents
    "https://www.gutenberg.org/files/66749/66749-0.txt", #The Fortunes of Perkin Warbeck: a romance by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/63337/63337-0.txt", #Valperga Volume 1 (of 3) by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/63338/63338-0.txt", #Valperga Volume 2 (of 3) by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/63339/63339-0.txt", #Valperga Volume 3 (of 3) by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/64555/64555-0.txt", #Lodore, Vol. 1 (of 3) by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/64556/64556-0.txt", #Lodore, Vol. 2 (of 3) by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/64557/64557-0.txt", #Lodore, Vol. 3 (of 3) by Mary Wollstonecraft Shelley
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
    "https://www.gutenberg.org/files/55012/55012-0.txt", #Dred: A Tale of the Great Dismal Swamp by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/47874/47874-0.txt", #My Wife and I; Or, Harry Henderson's History by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/31522/31522-0.txt", #The Pearl of Orr's Island: A Story of the Coast of Maine by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/47958/47958-0.txt", #The Minister's Wooing by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/43076/43076-0.txt", #Agnes of Sorrento by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/48603/48603-0.txt", #We and Our Neighbors; or, The Records of an Unfashionable Street by Stowe
    "https://www.gutenberg.org/files/45415/45415-0.txt", #The Chimney-Corner by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/67383/67383-0.txt", #Little Foxes by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/48190/48190-0.txt", #Poganuc People: Their Loves and Lives by Harriet Beecher Stowe
    "https://www.gutenberg.org/files/5843/5843-0.txt",   #The Young Step-Mother; Or, A Chronicle of Mistakes by Charlotte M. Yonge
    "https://www.gutenberg.org/files/6331/6331-0.txt",   #The Pillars of the House; Or, Under Wode, Under Rode, Vol. 1 (of 2) by Yonge
    "https://www.gutenberg.org/files/5124/5124-0.txt",   #Henrietta's Wish; Or, Domineering by Charlotte M. Yonge
    "https://www.gutenberg.org/files/44721/44721-0.txt", #The Pillars of the House; Or, Under Wode, Under Rode, Vol. 2 (of 2) by Yonge
    "https://www.gutenberg.org/files/5274/5274-0.txt",   #The Chaplet of Pearls by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4235/4235-0.txt",   #Dynevor Terrace; Or, The Clue of Life — Volume 1 by Charlotte M. Yonge
    "https://www.gutenberg.org/files/3744/3744-0.txt",   #The Trial; Or, More Links of the Daisy Chain by Charlotte M. Yonge
    "https://www.gutenberg.org/files/5708/5708-0.txt",   #Stray Pearls: Memoirs of Margaret De Ribaumont, Viscountess of Bellaise by Yonge
    "https://www.gutenberg.org/files/32200/32200-0.txt", #Sowing and Sewing: A Sexagesima Story by Charlotte M. Yonge
    "https://www.gutenberg.org/files/5251/5251-0.txt",   #The Long Vacation by Charlotte M. Yonge
    "https://www.gutenberg.org/files/6007/6007-0.txt",   #The Two Sides of the Shield by Charlotte M. Yonge
    "https://www.gutenberg.org/files/12449/12449-0.txt", #A Reputed Changeling; Or, Three Seventh Years Two Centuries Ago by Yonge
    "https://www.gutenberg.org/files/4236/4236-0.txt",   #Dynevor Terrace; Or, The Clue of Life — Volume 2 by Charlotte M. Yonge
    "https://www.gutenberg.org/files/9959/9959-0.txt",   #The Armourer's Prentices by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4364/4364-0.txt",   #The Lances of Lynwood by Charlotte M. Yonge
    "https://www.gutenberg.org/files/3139/3139-0.txt",   #The Dove in the Eagle's Nest by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4267/4267-0.txt",   #Abbeychurch; Or, Self-Control and Self-Conceit by Charlotte M. Yonge
    "https://www.gutenberg.org/files/2573/2573-0.txt",   #The Caged Lion by Charlotte M. Yonge
    "https://www.gutenberg.org/files/5156/5156-0.txt",   #Beechcroft at Rockstone by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4271/4271-0.txt",   #A Modern Telemachus by Charlotte M. Yonge
    "https://www.gutenberg.org/files/5313/5313-0.txt",   #The Herd Boy and His Hermit by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4296/4296-0.txt",   #Friarswood Post Office by Charlotte M. Yonge
    "https://www.gutenberg.org/files/21223/21223-0.txt", #The Carbonels by Charlotte M. Yonge
    "https://www.gutenberg.org/files/7387/7387-0.txt",   #Grisly Grisell; Or, The Laidly Lady of Whitburn: A Tale of the Wars of the…
    "https://www.gutenberg.org/files/20323/20323-0.txt", #That Stick by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4659/4659-0.txt",   #Lady Hester; Or, Ursula's Narrative by Charlotte M. Yonge
    "https://www.gutenberg.org/files/9926/9926-0.txt",   #The Two Guardians by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4053/4053-0.txt",   #Nuttie's Father by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4944/4944-0.txt",   #Scenes and Characters, or, Eighteen Months at Beechcroft by Charlotte M. Yonge
    "https://www.gutenberg.org/files/12485/12485-0.txt", #The Three Brides by Charlotte M. Yonge
    "https://www.gutenberg.org/files/3696/3696-0.txt",   #The Prince and the Page: A Story of the Last Crusade by Charlotte M. Yonge
    "https://www.gutenberg.org/files/4347/4347-0.txt",   #My Young Alcides: A Faded Photograph by Charlotte M. Yonge
    "https://www.gutenberg.org/files/24651/24651-0.txt", #Old Times at Otterbourne by Charlotte M. Yonge
    "https://www.gutenberg.org/files/25493/25493-0.txt", #A Cathedral Courtship by Kate Douglas Smith Wiggin
    "https://www.gutenberg.org/files/721/721-0.txt",     #The Birds' Christmas Carol by Kate Douglas Smith Wiggin
    "https://www.gutenberg.org/files/18531/18531-0.txt", #Timothy's Quest by Kate Douglas Smith Wiggin
    "https://www.gutenberg.org/files/15630/15630-0.txt", #Polly Oliver's Problem by Kate Douglas Smith Wiggin
    "https://www.gutenberg.org/files/13506/13506-0.txt", #The Story of Patsy by Kate Douglas Smith Wiggin
    "https://www.gutenberg.org/files/28877/28877-0.txt", #Penelope's Progress by Kate Douglas Smith Wiggin
    "https://www.gutenberg.org/files/28006/28006-0.txt", #The Perpetual Curate by Mrs. Oliphant
    "https://www.gutenberg.org/files/10052/10052-0.txt", #The Open Door, and the Portrait. by Mrs. Oliphant
    "https://www.gutenberg.org/files/41286/41286-0.txt", #Miss Marjoribanks by Mrs. Oliphant
    "https://www.gutenberg.org/files/28629/28629-0.txt", #Phoebe, Junior by Mrs. Oliphant
    "https://www.gutenberg.org/files/48197/48197-0.txt", #Hester: A Story of Contemporary Life, Volume 1 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/48199/48199-0.txt", #Hester: A Story of Contemporary Life, Volume 3 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/55868/55868-0.txt", #Lady William by Mrs. Oliphant
    "https://www.gutenberg.org/files/54106/54106-0.txt", #Neighbours on the Green by Mrs. Oliphant
    "https://www.gutenberg.org/files/59918/59918-0.txt", #The Story of a Governess by Mrs. Oliphant
    "https://www.gutenberg.org/files/42045/42045-0.txt", #The Curate in Charge by Mrs. Oliphant
    "https://www.gutenberg.org/files/11521/11521-0.txt", #A Beleaguered City by Mrs. Oliphant
    "https://www.gutenberg.org/files/42013/42013-0.txt", #Salem Chapel, v. 1/2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/54108/54108-0.txt", #Squire Arden; volume 1 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/47619/47619-0.txt", #At His Gates: A Novel. Vol. 2 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/45816/45816-0.txt", #The House on the Moor, v. 1/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/63902/63902-0.txt", #The Duke's Daughter; and, The Fugitives; vol. 2/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/28637/28637-0.txt", #The Marriage of Elinor by Mrs. Oliphant
    "https://www.gutenberg.org/files/55122/55122-0.txt", #The Athelings; or, the Three Gifts. Complete by Mrs. Oliphant
    "https://www.gutenberg.org/files/51265/51265-0.txt", #The Last of the Mortimers: A Story in Two Voices by Mrs. Oliphant
    "https://www.gutenberg.org/files/53744/53744-0.txt", #Sir Robert's Fortune: A Novel by Mrs. Oliphant
    "https://www.gutenberg.org/files/71919/71919-0.txt", #Christian Melville by Mrs. Oliphant
    "https://www.gutenberg.org/files/53583/53583-0.txt", #Ombra by Mrs. Oliphant
    "https://www.gutenberg.org/files/55827/55827-0.txt", #Who Was Lost and Is Found: A Novel by Mrs. Oliphant
    "https://www.gutenberg.org/files/45818/45818-0.txt", #The House on the Moor, v. 3/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/44080/44080-0.txt", #Madonna Mary by Mrs. Oliphant
    "https://www.gutenberg.org/files/47555/47555-0.txt", #The Wizard's Son, Vol. 1 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/54122/54122-0.txt", #Squire Arden; volume 2 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/55155/55155-0.txt", #Old Mr. Tredgold by Mrs. Oliphant
    "https://www.gutenberg.org/files/52575/52575-0.txt", #The Cuckoo in the Nest, v. 1/2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/47591/47591-0.txt", #The Ladies Lindores, Vol. 1 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/48198/48198-0.txt", #Hester: A Story of Contemporary Life, Volume 2 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/52060/52060-0.txt", #The Sorceress (complete) by Mrs. Oliphant
    "https://www.gutenberg.org/files/63562/63562-0.txt", #Harry Joscelyn; vol. 3 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/65329/65329-0.txt", #Mrs. Arthur; vol. 2 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/65153/65153-0.txt", #Within the Precincts by Mrs. Oliphant
    "https://www.gutenberg.org/files/65328/65328-0.txt", #Mrs. Arthur; vol. 1 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/72221/72221-0.txt", #Carità by Mrs. Oliphant
    "https://www.gutenberg.org/files/58462/58462-0.txt", #The Story of Valentine and His Brother by Mrs. Oliphant
    "https://www.gutenberg.org/files/49597/49597-0.txt", #It was a Lover and His Lass by Mrs. Oliphant
    "https://www.gutenberg.org/files/47620/47620-0.txt", #At His Gates: A Novel. Vol. 3 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/47557/47557-0.txt", #The Wizard's Son, Vol. 3 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/64540/64540-0.txt", #The Greatest Heiress in England by Mrs. Oliphant
    "https://www.gutenberg.org/files/61916/61916-0.txt", #Effie Ogilvie: the story of a young life (Complete) by Mrs. Oliphant
    "https://www.gutenberg.org/files/53182/53182-0.txt", #The Sorceress, v. 3 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/52388/52388-0.txt", #Whiteladies by Mrs. Oliphant
    "https://www.gutenberg.org/files/10050/10050-0.txt", #A Little Pilgrim: Stories of the Seen and the Unseen by Mrs. Oliphant
    "https://www.gutenberg.org/files/65226/65226-0.txt", #John, A Love Story; vol. 1 of 2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/29890/29890-0.txt", #The Doctor's Family by Mrs. Oliphant
    "https://www.gutenberg.org/files/61954/61954-0.txt", #The Lady's Walk by Mrs. Oliphant
    "https://www.gutenberg.org/files/63142/63142-0.txt", #Harry Joscelyn; vol. 1 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/64778/64778-0.txt", #He that will not when he may; vol. II by Mrs. Oliphant
    "https://www.gutenberg.org/files/59055/59055-0.txt", #The Heir Presumptive and the Heir Apparent by Mrs. Oliphant
    "https://www.gutenberg.org/files/54841/54841-0.txt", #Brownlows: A Novel by Mrs. Oliphant
    "https://www.gutenberg.org/files/59759/59759-0.txt", #The Executor by Mrs. Oliphant
    "https://www.gutenberg.org/files/71671/71671-0.txt", #Innocent : a tale of modern life by Mrs. Oliphant
    "https://www.gutenberg.org/files/65631/65631-0.txt", #May; vol. I by Mrs. Oliphant
    "https://www.gutenberg.org/files/65458/65458-0.txt", #The Primrose Path: A Chapter in the Annals of the Kingdom of Fife by Mrs. Oliphant
    "https://www.gutenberg.org/files/62466/62466-0.txt", #The Prodigals and Their Inheritance; Complete by Mrs. Oliphant
    "https://www.gutenberg.org/files/52756/52756-0.txt", #The Cuckoo in the Nest, v. 2/2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/61445/61445-0.txt", #A House Divided Against Itself (Complete) by Mrs. Oliphant
    "https://www.gutenberg.org/files/58710/58710-0.txt", #The Land of Darkness by Mrs. Oliphant
    "https://www.gutenberg.org/files/53645/53645-0.txt", #Heart and Cross by Mrs. Oliphant
    "https://www.gutenberg.org/files/66376/66376-0.txt", #Young Musgrave by Mrs. Oliphant
    "https://www.gutenberg.org/files/63903/63903-0.txt", #The Duke's Daughter; and, The Fugitives; vol. 3/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/30692/30692-0.txt", #Sir Tom by Mrs. Oliphant (corrected Gutenberg ID)
    "https://www.gutenberg.org/files/47618/47618-0.txt", #At His Gates: A Novel. Vol. 1 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/74734/74734-0.txt", #Caleb Field : A tale of the Puritans by Mrs. Oliphant
    "https://www.gutenberg.org/files/61782/61782-0.txt", #A Poor Gentleman by Mrs. Oliphant
    "https://www.gutenberg.org/files/47592/47592-0.txt", #The Ladies Lindores, Vol. 2 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/55784/55784-0.txt", #Two Strangers by Mrs. Oliphant
    "https://www.gutenberg.org/files/47593/47593-0.txt", #The Ladies Lindores, Vol. 3 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/63158/63158-0.txt", #Harry Joscelyn; vol. 2 of 3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/58707/58707-0.txt", #Kirsteen: The Story of a Scotch Family Seventy Years Ago by Mrs. Oliphant
    "https://www.gutenberg.org/files/45817/45817-0.txt", #The House on the Moor, v. 2/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/58253/58253-0.txt", #The Minister's Wife by Mrs. Oliphant
    "https://www.gutenberg.org/files/64779/64779-0.txt", #He that will not when he may; vol. III by Mrs. Oliphant
    "https://www.gutenberg.org/files/59720/59720-0.txt", #The Son of His Father; vol. 1/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/55166/55166-0.txt", #The Unjust Steward; or, The Minister's Debt by Mrs. Oliphant
    "https://www.gutenberg.org/files/52615/52615-0.txt", #The Two Marys by Mrs. Oliphant
    "https://www.gutenberg.org/files/65633/65633-0.txt", #May; vol. II by Mrs. Oliphant
    "https://www.gutenberg.org/files/55140/55140-0.txt", #A House in Bloomsbury by Mrs. Oliphant
    "https://www.gutenberg.org/files/63302/63302-0.txt", #Cousin Mary by Mrs. Oliphant
    "https://www.gutenberg.org/files/59721/59721-0.txt", #The Son of His Father; vol. 2/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/59424/59424-0.txt", #The Railway Man and His Children by Mrs. Oliphant
    "https://www.gutenberg.org/files/47556/47556-0.txt", #The Wizard's Son, Vol. 2 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/61482/61482-0.txt", #The Mystery of Mrs. Blencarrow by Mrs. Oliphant
    "https://www.gutenberg.org/files/65227/65227-0.txt", #John, A Love Story; vol. 2 of 2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/58323/58323-0.txt", #The Three Brothers; Complete by Mrs. Oliphant
    "https://www.gutenberg.org/files/74930/74930-0.txt", #Memoirs and resolutions of Adam Graeme of Mossgray, including some chronicles…
    "https://www.gutenberg.org/files/57988/57988-0.txt", #A son of the soil by Mrs. Oliphant
    "https://www.gutenberg.org/files/63901/63901-0.txt", #The Duke's Daughter; and, The Fugitives; vol. 1/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/64777/64777-0.txt", #He that will not when he may; vol. I by Mrs. Oliphant
    "https://www.gutenberg.org/files/10051/10051-0.txt", #The Little Pilgrim: Further Experiences. by Mrs. Oliphant
    "https://www.gutenberg.org/files/65935/65935-0.txt", #For love and life; vol. 2 of 2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/15410/15410-0.txt", #A Little Pilgrim in the Unseen by Mrs. Oliphant
    "https://www.gutenberg.org/files/60018/60018-0.txt", #The Son of His Father; vol. 3/3 by Mrs. Oliphant
    "https://www.gutenberg.org/files/63582/63582-0.txt", #Oliver's Bride; A true Story by Mrs. Oliphant
    "https://www.gutenberg.org/files/59326/59326-0.txt", #The Quiet Heart by Mrs. Oliphant
    "https://www.gutenberg.org/files/65934/65934-0.txt", #For love and life; vol. 1 of 2 by Mrs. Oliphant
    "https://www.gutenberg.org/files/73086/73086-0.txt", #Harry Muir : A story of Scottish life, vol. 3 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/72777/72777-0.txt", #Harry Muir : A story of Scottish life, vol. 2 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/72715/72715-0.txt", #Harry Muir : A story of Scottish life, vol. 1 (of 3) by Mrs. Oliphant
    "https://www.gutenberg.org/files/78381/78381-0.txt", #An odd couple by Mrs. Oliphant
    "https://www.gutenberg.org/files/58459/58459-0.txt", #Lady Car: The Sequel of a Life by Mrs. Oliphant
    "https://www.gutenberg.org/files/58446/58446-0.txt", #Sons and Daughters by Mrs. Oliphant
    "https://www.gutenberg.org/files/29891/29891-0.txt", #The Rector by Mrs. Oliphant
    "https://www.gutenberg.org/files/24898/24898-0.txt", #Robert Elsmere by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/8076/8076-0.txt",   #The History of David Grieve by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/13728/13728-0.txt", #Marcella by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/9633/9633-0.txt",   #Sir George Tressady — Volume I by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/9634/9634-0.txt",   #Sir George Tressady — Volume II by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/9441/9441-0.txt",   #Helbeck of Bannisdale — Volume I by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/9442/9442-0.txt",   #Helbeck of Bannisdale — Volume II by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/13432/13432-0.txt", #Miss Bretherton by Mrs. Humphry Ward
    "https://www.gutenberg.org/files/287/287-0.txt",     #Remember the Alamo by Amelia E. Barr
    "https://www.gutenberg.org/files/16258/16258-0.txt", #The Squire of Sandal-Side: A Pastoral Romance by Amelia E. Barr
    "https://www.gutenberg.org/files/29909/29909-0.txt", #A Singer from the Sea by Amelia E. Barr
    "https://www.gutenberg.org/files/32144/32144-0.txt", #Jan Vedder's Wife by Amelia E. Barr
    "https://www.gutenberg.org/files/31835/31835-0.txt", #Prisoners of Conscience by Amelia E. Barr
    "https://www.gutenberg.org/files/17173/17173-0.txt", #The Bow of Orange Ribbon: A Romance of New York by Amelia E. Barr
    "https://www.gutenberg.org/files/7062/7062-0.txt",   #A Daughter of Fife by Amelia E. Barr
    "https://www.gutenberg.org/files/9374/9374-0.txt",   #A Knight of the Nets by Amelia E. Barr
    "https://www.gutenberg.org/files/6806/6806-0.txt",   #The Hallam Succession by Amelia E. Barr
    "https://www.gutenberg.org/files/33195/33195-0.txt", #Was It Right to Forgive? A Domestic Romance by Amelia E. Barr
    "https://www.gutenberg.org/files/34628/34628-0.txt", #I, Thou, and the Other One: A Love Story by Amelia E. Barr
    "https://www.gutenberg.org/files/33599/33599-0.txt", #A Rose of a Hundred Leaves: A Love Story by Amelia E. Barr
    "https://www.gutenberg.org/files/46363/46363-0.txt", #Her Dark Inheritance by Mrs. E. Burke Collins
    "https://www.gutenberg.org/files/70139/70139-0.txt", #A bitter reckoning; or, Violet Arleigh by Mrs. E. Burke Collins
    "https://www.gutenberg.org/files/53009/53009-0.txt", #Patience Sparhawk and Her Times: A Novel by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/3706/3706-0.txt",   #The Valiant Runaways by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/50169/50169-0.txt", #Hermia Suydam by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/35512/35512-0.txt", #A Daughter of the Vine by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/12833/12833-0.txt", #What Dreams May Come by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/67075/67075-0.txt", #A Whirl Asunder by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/68222/68222-0.txt", #His fortunate Grace by Gertrude Franklin Horn Atherton
    "https://www.gutenberg.org/files/51103/51103-0.txt", #A Chautauqua Idyl by Grace Livingston Hill
    "https://www.gutenberg.org/files/26742/26742-0.txt", #The Chautauqua Girls At Home by Pansy
    "https://www.gutenberg.org/files/20808/20808-0.txt", #Three People by Pansy
    "https://www.gutenberg.org/files/9091/9091-0.txt",   #Ester Ried Yet Speaking by Pansy
    "https://www.gutenberg.org/files/45537/45537-0.txt", #Interrupted by Pansy
    "https://www.gutenberg.org/files/54078/54078-0.txt", #Ruth Erskine's Crosses by Pansy
    "https://www.gutenberg.org/files/9648/9648-0.txt",   #Tip Lewis and His Lamp by Pansy
    "https://www.gutenberg.org/files/12662/12662-0.txt", #Four Girls at Chautauqua by Pansy
    "https://www.gutenberg.org/files/60174/60174-0.txt", #The Children of the Abbey: A Tale by Regina Maria Roche
    "https://www.gutenberg.org/files/17886/17886-0.txt", #Jerome, A Poor Man: A Novel by Mary Eleanor Wilkins Freeman
    "https://www.gutenberg.org/files/17428/17428-0.txt", #Pembroke: A Novel by Mary Eleanor Wilkins Freeman
    "https://www.gutenberg.org/files/17792/17792-0.txt", #The Jamesons by Mary Eleanor Wilkins Freeman
    "https://www.gutenberg.org/files/17885/17885-0.txt", #Madelon: A Novel by Mary Eleanor Wilkins Freeman
    "https://www.gutenberg.org/files/17891/17891-0.txt", #Evelina's Garden by Mary Eleanor Wilkins Freeman
    "https://www.gutenberg.org/files/17790/17790-0.txt", #Jane Field: A Novel by Mary Eleanor Wilkins Freeman
    "https://www.gutenberg.org/files/41655/41655-0.txt", #Stephen: A Soldier of the Cross by Florence Morse Kingsley
    "https://www.gutenberg.org/files/29581/29581-0.txt", #The Bondwoman by Marah Ellis Ryan
    "https://www.gutenberg.org/files/36246/36246-0.txt", #Told in the Hills: A Novel by Marah Ellis Ryan
    "https://www.gutenberg.org/files/2374/2374-0.txt",   #Dora Thorne by Charlotte M. Brame
    "https://www.gutenberg.org/files/33989/33989-0.txt", #Love Works Wonders: A Novel by Charlotte M. Brame
    "https://www.gutenberg.org/files/31489/31489-0.txt", #A Mad Love by Charlotte M. Brame
    "https://www.gutenberg.org/files/41932/41932-0.txt", #A Fair Mystery: The Story of a Coquette by Charlotte M. Brame
    "https://www.gutenberg.org/files/12886/12886-0.txt", #The Coquette's Victim by Charlotte M. Brame
    "https://www.gutenberg.org/files/13162/13162-0.txt", #Coralie by Charlotte M. Brame
    "https://www.gutenberg.org/files/15181/15181-0.txt", #My Mother's Rival by Charlotte M. Brame
    "https://www.gutenberg.org/files/15183/15183-0.txt", #The Tragedy of the Chain Pier by Charlotte M. Brame
    "https://www.gutenberg.org/files/15182/15182-0.txt", #Marion Arleigh's Penance by Charlotte M. Brame
    "https://www.gutenberg.org/files/12669/12669-0.txt", #Marriage by Susan Ferrier
    "https://www.gutenberg.org/files/46708/46708-0.txt", #The Dead Letter: An American Romance by Metta Victoria Fuller Victor
    "https://www.gutenberg.org/files/46586/46586-0.txt", #Alice Wilde: The Raftsman's Daughter. A Forest Romance by Victor
    "https://www.gutenberg.org/files/20754/20754-0.txt", #The Blunders of a Bashful Man by Metta Victoria Fuller Victor
    "https://www.gutenberg.org/files/68363/68363-0.txt", #Nat Wolfe; or, The gold hunters: A romance of Pike's Peak and New York by Victor
    "https://www.gutenberg.org/files/1327/1327-0.txt",   #Elizabeth and Her German Garden by Elizabeth Von Arnim
    "https://www.gutenberg.org/files/5991/5991-0.txt",   #The Solitary Summer by Elizabeth Von Arnim
    "https://www.gutenberg.org/files/31879/31879-0.txt", #Coelebs In Search of a Wife by Hannah More
    "https://www.gutenberg.org/files/23106/23106-0.txt", #Helen and Arthur; or, Miss Thusa's Spinning Wheel by Caroline Lee Hentz
    "https://www.gutenberg.org/files/20462/20462-0.txt", #Ernest Linwood; or, The Inner Life of the Author by Caroline Lee Hentz
    "https://www.gutenberg.org/files/21690/21690-0.txt", #Flint: His Faults, His Friendships and His Fortunes by Maud Wilder Goodwin
    "https://www.gutenberg.org/files/54066/54066-0.txt", #The Younger Sister: A Novel, Volumes 1-3 by Mrs. Hubback
    "https://www.gutenberg.org/files/70732/70732-0.txt", #The old vicarage : A novel by Mrs. Hubback
    "https://www.gutenberg.org/files/23780/23780-0.txt", #A Little Girl in Old New York by Amanda M. Douglas
    "https://www.gutenberg.org/files/23781/23781-0.txt", #A Little Girl of Long Ago; Or, Hannah Ann by Amanda M. Douglas
    "https://www.gutenberg.org/files/28648/28648-0.txt", #A Little Girl in Old Philadelphia by Amanda M. Douglas
    "https://www.gutenberg.org/files/23786/23786-0.txt", #A Little Girl in Old Boston by Amanda M. Douglas
    "https://www.gutenberg.org/files/30436/30436-0.txt", #Hope Mills; Or, Between Friend and Sweetheart by Amanda M. Douglas
    "https://www.gutenberg.org/files/43659/43659-0.txt", #The Old Woman Who Lived in a Shoe; Or, There's No Place Like Home by Douglas
    "https://www.gutenberg.org/files/24376/24376-0.txt", #Floyd Grandon's Honor by Amanda M. Douglas
    "https://www.gutenberg.org/files/36907/36907-0.txt", #In Wild Rose Time by Amanda M. Douglas
    "https://www.gutenberg.org/files/40525/40525-0.txt", #Kathie's Soldiers by Amanda M. Douglas
    "https://www.gutenberg.org/files/71729/71729-0.txt", #Seven daughters by Amanda M. Douglas
    "https://www.gutenberg.org/files/25551/25551-0.txt", #Six Girls: A Home Story by Fannie Belle Irving
    "https://www.gutenberg.org/files/134/134-0.txt",     #Maria; Or, The Wrongs of Woman by Mary Wollstonecraft
    "https://www.gutenberg.org/files/16357/16357-0.txt", #Mary: A Fiction by Mary Wollstonecraft
    "https://www.gutenberg.org/files/37355/37355-0.txt", #Pray You, Sir, Whose Daughter? by Helen H. Gardener
    "https://www.gutenberg.org/files/42328/42328-0.txt", #An Unofficial Patriot by Helen H. Gardener
    "https://www.gutenberg.org/files/25481/25481-0.txt", #Little Grandfather by Sophie May
    "https://www.gutenberg.org/files/23540/23540-0.txt", #The Twin Cousins by Sophie May
    "https://www.gutenberg.org/files/25507/25507-0.txt", #Little Grandmother by Sophie May
    "https://www.gutenberg.org/files/25356/25356-0.txt", #Aunt Madge's Story by Sophie May
    "https://www.gutenberg.org/files/25484/25484-0.txt", #Captain Horace by Sophie May
    "https://www.gutenberg.org/files/20984/20984-0.txt", #Prudy Keeping House by Sophie May
    "https://www.gutenberg.org/files/24711/24711-0.txt", #Little Prudy by Sophie May
    "https://www.gutenberg.org/files/48648/48648-0.txt", #Doctor Papa by Sophie May
    "https://www.gutenberg.org/files/20699/20699-0.txt", #Dotty Dimple at Her Grandmother's by Sophie May
    "https://www.gutenberg.org/files/11257/11257-0.txt", #Little Folks Astray by Sophie May
    "https://www.gutenberg.org/files/19247/19247-0.txt", #Dotty Dimple's Flyaway by Sophie May
    "https://www.gutenberg.org/files/48951/48951-0.txt", #Dotty Dimple at School by Sophie May
    "https://www.gutenberg.org/files/25396/25396-0.txt", #Dotty Dimple At Home by Sophie May
    "https://www.gutenberg.org/files/49227/49227-0.txt", #Kittyleen by Sophie May
    "https://www.gutenberg.org/files/14202/14202-0.txt", #Little Prudy's Sister Susy by Sophie May
    "https://www.gutenberg.org/files/49519/49519-0.txt", #Jimmy Boy by Sophie May
    "https://www.gutenberg.org/files/16390/16390-0.txt", #Little Prudy's Dotty Dimple by Sophie May
    "https://www.gutenberg.org/files/48862/48862-0.txt", #Little Pitchers by Sophie May
    "https://www.gutenberg.org/files/49186/49186-0.txt", #Flaxie Growing Up by Sophie May
    "https://www.gutenberg.org/files/16383/16383-0.txt", #Dotty Dimple Out West by Sophie May
    "https://www.gutenberg.org/files/49686/49686-0.txt", #Little Prudy's Cousin Grace by Sophie May
    "https://www.gutenberg.org/files/10320/10320-0.txt", #Dotty Dimple at Play by Sophie May
    "https://www.gutenberg.org/files/367/367-0.txt",     #The Country of the Pointed Firs by Sarah Orne Jewett
    "https://www.gutenberg.org/files/27923/27923-0.txt", #Betty Leicester: A Story For Girls by Sarah Orne Jewett
    "https://www.gutenberg.org/files/41831/41831-0.txt", #Betty Leicester's Christmas by Sarah Orne Jewett
    "https://www.gutenberg.org/files/23795/23795-0.txt", #Katie Robertson : A girls story of factory life by Margaret E. Winslow
    "https://www.gutenberg.org/files/335/335-0.txt",     #Frances Waldeaux: A Novel by Rebecca Harding Davis
    "https://www.gutenberg.org/files/515/515-0.txt",     #Margret Howth: A Story of To-day by Rebecca Harding Davis
    "https://www.gutenberg.org/files/27811/27811-0.txt", #Macaria by Augusta J. Evans
    "https://www.gutenberg.org/files/4209/4209-0.txt",   #At the Mercy of Tiberius by Augusta J. Evans
    "https://www.gutenberg.org/files/31620/31620-0.txt", #Vashti; Or, Until Death Us Do Part by Augusta J. Evans
    "https://www.gutenberg.org/files/4553/4553-0.txt",   #St. Elmo by Augusta J. Evans
    "https://www.gutenberg.org/files/4246/4246-0.txt",   #Beulah by Augusta J. Evans
    "https://www.gutenberg.org/files/15470/15470-0.txt", #Inez: A Tale of the Alamo by Augusta J. Evans
    "https://www.gutenberg.org/files/2807/2807-0.txt",   #To Have and to Hold by Mary Johnston
    "https://www.gutenberg.org/files/21886/21886-0.txt", #Prisoners of Hope: A Tale of Colonial Virginia by Mary Johnston
    "https://www.gutenberg.org/files/25118/25118-0.txt", #When Grandmamma Was New: The Story of a Virginia Childhood by Marion Harland
    "https://www.gutenberg.org/files/46505/46505-0.txt", #Alone by Marion Harland
    "https://www.gutenberg.org/files/5622/5622-0.txt",   #At Last: A Novel by Marion Harland
    "https://www.gutenberg.org/files/36414/36414-0.txt", #Jessamine: A Novel by Marion Harland
    "https://www.gutenberg.org/files/48461/48461-0.txt", #Husks by Marion Harland
    "https://www.gutenberg.org/files/50512/50512-0.txt", #Mr. Wayt's Wife's Sister by Marion Harland
    "https://www.gutenberg.org/files/64117/64117-0.txt", #The Christmas Holly by Marion Harland
    "https://www.gutenberg.org/files/69636/69636-0.txt", #Salome Shepard, reformer by Helen M. Winslow
    "https://www.gutenberg.org/files/65290/65290-0.txt", #Among the Lindens by Evelyn Raymond
    "https://www.gutenberg.org/files/67567/67567-0.txt", #The Little Lady of the Horse by Evelyn Raymond
    "https://www.gutenberg.org/files/65223/65223-0.txt", #Mixed Pickles by Evelyn Raymond
    "https://www.gutenberg.org/files/15546/15546-0.txt", #The Last of the Peterkins by Lucretia P. Hale
    "https://www.gutenberg.org/files/22052/22052-0.txt", #Standish of Standish: A Story of the Pilgrims by Jane G. Austin
    "https://www.gutenberg.org/files/67608/67608-0.txt", #Betty Alden: The first-born daughter of the Pilgrims by Jane G. Austin
    "https://www.gutenberg.org/files/4676/4676-0.txt",   #Outpost by Jane G. Austin
    "https://www.gutenberg.org/files/24828/24828-0.txt", #Margaret Montfort by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/24826/24826-0.txt", #Hildegarde's Holiday: A Story for Girls by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/22293/22293-0.txt", #Three Margarets by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/25448/25448-0.txt", #Peggy by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/27607/27607-0.txt", #Rosin the Beau by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/16473/16473-0.txt", #Queen Hildegarde by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/41603/41603-0.txt", #Toto's Merry Winter by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/35281/35281-0.txt", #The Joyous Story of Toto by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/7824/7824-0.txt",   #Melody : The Story of a Child by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/34218/34218-0.txt", #Hildegarde's Home by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/27251/27251-0.txt", #"Some Say"; Neighbours in Cyrus by Laura Elizabeth Howe Richards
    "https://www.gutenberg.org/files/23859/23859-0.txt", #A Venetian June by Anna Fuller
    "https://www.gutenberg.org/files/27949/27949-0.txt", #Daisy by Susan Warner
    "https://www.gutenberg.org/files/28376/28376-0.txt", #The Wide, Wide World by Susan Warner
    "https://www.gutenberg.org/files/8874/8874-0.txt",   #Queechy by Susan Warner
    "https://www.gutenberg.org/files/36159/36159-0.txt", #The Letter of Credit by Susan Warner
    "https://www.gutenberg.org/files/27618/27618-0.txt", #The End of a Coil by Susan Warner
    "https://www.gutenberg.org/files/26828/26828-0.txt", #A Red Wallflower by Susan Warner
    "https://www.gutenberg.org/files/30148/30148-0.txt", #The House in Town by Susan Warner
    "https://www.gutenberg.org/files/29824/29824-0.txt", #Diana by Susan Warner
    "https://www.gutenberg.org/files/18689/18689-0.txt", #The Wide, Wide World by Susan Warner
    "https://www.gutenberg.org/files/12963/12963-0.txt", #Melbourne House, Volume 1 by Susan Warner
    "https://www.gutenberg.org/files/30146/30146-0.txt", #What She Could by Susan Warner
    "https://www.gutenberg.org/files/28524/28524-0.txt", #Nobody by Susan Warner
    "https://www.gutenberg.org/files/30149/30149-0.txt", #Trading by Susan Warner
    "https://www.gutenberg.org/files/16918/16918-0.txt", #Hills of the Shatemuc by Susan Warner
    "https://www.gutenberg.org/files/8954/8954-0.txt",   #Lady Audley's Secret by M. E. Braddon
    "https://www.gutenberg.org/files/75840/75840-0.txt", #The trail of the serpent by M. E. Braddon
    "https://www.gutenberg.org/files/11720/11720-0.txt", #Fenton's Quest by M. E. Braddon
    "https://www.gutenberg.org/files/9189/9189-0.txt",   #Henry Dunbar: A Novel by M. E. Braddon
    "https://www.gutenberg.org/files/35485/35485-0.txt", #The Doctor's Wife: A Novel by M. E. Braddon
    "https://www.gutenberg.org/files/9475/9475-0.txt",   #The Lovels of Arden by M. E. Braddon
    "https://www.gutenberg.org/files/50037/50037-0.txt", #Wyllard's Weird: A Novel by M. E. Braddon
    "https://www.gutenberg.org/files/70938/70938-0.txt", #The day will come : a novel by M. E. Braddon
    "https://www.gutenberg.org/files/48020/48020-0.txt", #Aurora Floyd, Vol. 1 by M. E. Braddon
    "https://www.gutenberg.org/files/48021/48021-0.txt", #Aurora Floyd, Vol. 2 by M. E. Braddon
    "https://www.gutenberg.org/files/48022/48022-0.txt", #Aurora Floyd, Vol. 3 by M. E. Braddon
    "https://www.gutenberg.org/files/57405/57405-0.txt", #All along the River: A Novel by M. E. Braddon
    "https://www.gutenberg.org/files/10905/10905-0.txt", #Phantom Fortune, a Novel by M. E. Braddon
    "https://www.gutenberg.org/files/74385/74385-0.txt", #The Venetians : A novel by M. E. Braddon
    "https://www.gutenberg.org/files/9052/9052-0.txt",   #The Golden Calf by M. E. Braddon
    "https://www.gutenberg.org/files/9377/9377-0.txt",   #London Pride, Or, When the World Was Younger by M. E. Braddon
    "https://www.gutenberg.org/files/34542/34542-0.txt", #John Marchmont's Legacy, Volumes 1-3 by M. E. Braddon
    "https://www.gutenberg.org/files/73931/73931-0.txt", #The cloven foot : A novel by M. E. Braddon
    "https://www.gutenberg.org/files/74484/74484-0.txt", #The Octoroon by M. E. Braddon
    "https://www.gutenberg.org/files/9362/9362-0.txt",   #Birds of Prey by M. E. Braddon
    "https://www.gutenberg.org/files/76927/76927-0.txt", #Only a clod by M. E. Braddon
    "https://www.gutenberg.org/files/9102/9102-0.txt",   #Run to Earth: A Novel by M. E. Braddon
    "https://www.gutenberg.org/files/26236/26236-0.txt", #Vixen, Volume I. by M. E. Braddon
    "https://www.gutenberg.org/files/26237/26237-0.txt", #Vixen, Volume II. by M. E. Braddon
    "https://www.gutenberg.org/files/26238/26238-0.txt", #Vixen, Volume III. by M. E. Braddon
    "https://www.gutenberg.org/files/75125/75125-0.txt", #Eleanor's victory by M. E. Braddon
    "https://www.gutenberg.org/files/71631/71631-0.txt", #His darling sin by M. E. Braddon
    "https://www.gutenberg.org/files/76502/76502-0.txt", #The lady's mile : a novel by M. E. Braddon
    "https://www.gutenberg.org/files/9259/9259-0.txt",   #Charlotte's Inheritance by M. E. Braddon
    "https://www.gutenberg.org/files/75410/75410-0.txt", #The fatal three, vol. I by M. E. Braddon
    "https://www.gutenberg.org/files/75411/75411-0.txt", #The fatal three, vol. II by M. E. Braddon
    "https://www.gutenberg.org/files/75412/75412-0.txt", #The fatal three, vol. III by M. E. Braddon
    "https://www.gutenberg.org/files/75506/75506-0.txt", #Asphodel : a novel by M. E. Braddon
    "https://www.gutenberg.org/files/75875/75875-0.txt", #Lucius Davoren; or, Publicans and sinners, vol. 1 by M. E. Braddon
    "https://www.gutenberg.org/files/75876/75876-0.txt", #Lucius Davoren; or, Publicans and sinners, vol. 2 by M. E. Braddon
    "https://www.gutenberg.org/files/75877/75877-0.txt", #Lucius Davoren; or, Publicans and sinners, vol. 3 by M. E. Braddon
    "https://www.gutenberg.org/files/77631/77631-0.txt", #Rupert Godwin : A novel by M. E. Braddon
    "https://www.gutenberg.org/files/41339/41339-0.txt", #Mount Royal: A Novel. Volume 1 of 3 by M. E. Braddon
    "https://www.gutenberg.org/files/67237/67237-0.txt", #An open verdict : a novel, volume 1 (of 3) by M. E. Braddon
    "https://www.gutenberg.org/files/68040/68040-0.txt", #An open verdict : a novel, volume 3 (of 3) by M. E. Braddon
    "https://www.gutenberg.org/files/57086/57086-0.txt", #A Strange World: A Novel. Volume 1 (of 3) by M. E. Braddon
    "https://www.gutenberg.org/files/41374/41374-0.txt", #Mohawks: A Novel. Volume 1 of 3 by M. E. Braddon
    "https://www.gutenberg.org/files/41375/41375-0.txt", #Mohawks: A Novel. Volume 2 of 3 by M. E. Braddon
    "https://www.gutenberg.org/files/23641/23641-0.txt", #The Forsaken Inn: A Novel by Anna Katharine Green
    "https://www.gutenberg.org/files/4047/4047-0.txt",   #The Leavenworth Case by Anna Katharine Green
    "https://www.gutenberg.org/files/21617/21617-0.txt", #That Affair Next Door by Anna Katharine Green
    "https://www.gutenberg.org/files/1167/1167-0.txt",   #A Strange Disappearance by Anna Katharine Green
    "https://www.gutenberg.org/files/33301/33301-0.txt", #The Sword of Damocles: A Story of New York Life by Anna Katharine Green
    "https://www.gutenberg.org/files/31681/31681-0.txt", #Hand and Ring by Anna Katharine Green
    "https://www.gutenberg.org/files/5162/5162-0.txt",   #Agatha Webb by Anna Katharine Green
    "https://www.gutenberg.org/files/33305/33305-0.txt", #Lost Man's Lane: A Second Episode in the Life of Amelia Butterworth by Green
    "https://www.gutenberg.org/files/32439/32439-0.txt", #The Doctor, his Wife, and the Clock by Anna Katharine Green
    "https://www.gutenberg.org/files/33695/33695-0.txt", #X Y Z: A Detective Story by Anna Katharine Green
    "https://www.gutenberg.org/files/36758/36758-0.txt", #Cynthia Wakeham's Money by Anna Katharine Green
    "https://www.gutenberg.org/files/22809/22809-0.txt", #The Hermit Of ——— Street by Anna Katharine Green
    "https://www.gutenberg.org/files/56787/56787-0.txt", #7 to 12: A Detective Story by Anna Katharine Green
    "https://www.gutenberg.org/files/57643/57643-0.txt", #Doctor Izard by Anna Katharine Green
    "https://www.gutenberg.org/files/6805/6805-0.txt",   #The Mill Mystery by Anna Katharine Green
    "https://www.gutenberg.org/files/22806/22806-0.txt", #The Bronze Hand by Anna Katharine Green
    "https://www.gutenberg.org/files/24750/24750-0.txt", #Mizora: A Prophecy by Mary E. Bradley Lane
    "https://www.gutenberg.org/files/770/770-0.txt",     #The Story of the Treasure Seekers by E. Nesbit
    "https://www.gutenberg.org/files/40814/40814-0.txt", #Ruth Hall: A Domestic Tale of the Present Time by Fanny Fern
    "https://www.gutenberg.org/files/41542/41542-0.txt", #Rose Clark by Fanny Fern
    "https://www.gutenberg.org/files/1905/1905-0.txt",   #The Governess; Or, The Little Female Academy by Sarah Fielding
    "https://www.gutenberg.org/files/79731/79731-0.txt", #On the edge of a moor by Amy Le Feuvre
    "https://www.gutenberg.org/files/22243/22243-0.txt", #Dwell Deep; or, Hilda Thorn's Life Story by Amy Le Feuvre
    "https://www.gutenberg.org/files/11470/11470-0.txt", #His big opportunity by Amy Le Feuvre
    "https://www.gutenberg.org/files/22232/22232-0.txt", #The Carved Cupboard by Amy Le Feuvre
    "https://www.gutenberg.org/files/10777/10777-0.txt", #Probable Sons by Amy Le Feuvre
    "https://www.gutenberg.org/files/76174/76174-0.txt", #Legend-led by Amy Le Feuvre
    "https://www.gutenberg.org/files/72454/72454-0.txt", #What the wind did by Amy Le Feuvre
    "https://www.gutenberg.org/files/10880/10880-0.txt", #Teddy's Button by Amy Le Feuvre
    "https://www.gutenberg.org/files/35538/35538-0.txt", #The Squatter and the Don by María Amparo Ruiz de Burton
    "https://www.gutenberg.org/files/79593/79593-0.txt", #Work, wait, win by Ruth Lamb


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


def split_for(book_id: str, seed: str, validation: float, test: float) -> str:
    """Assign each Gutenberg edition independently, stable as the URL list grows."""
    value = int(hashlib.sha256(f"{seed}\0{book_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value < test else "validation" if value < test + validation else "train"


VERSION = "3.1"
DEFAULT_MIRROR = "https://gutenberg.pglaf.org"
# Long ebook titles can wrap onto up to two continuation lines inside the marker.
START = re.compile(r"(?im)^[ \t]*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT)\b[^\n]*(?:\n[^\n*]{1,120}){0,2}\*{3}[^\n]*$")
END = re.compile(r"(?im)^[ \t]*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT)\b[^\n]*(?:\n[^\n*]{1,120}){0,2}\*{3}[^\n]*$")
LEGACY_END = re.compile(r"(?im)^[ \t]*End of (?:the )?Project Gutenberg(?:[’']s)?(?: (?:EBook|Etext))?\b[^\n]*$")


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
    modern_ends = list(END.finditer(text))
    ends = modern_ends or list(LEGACY_END.finditer(text))
    if len(starts) != 1 or len(ends) != 1 or ends[0].start() <= starts[0].end():
        raise CorpusError("Missing, ambiguous, or reversed Gutenberg start/end markers; refusing to guess.")
    start, end = starts[0], ends[0]
    # Older editions sometimes contain both footer styles. Strip the inner one too.
    legacy = [m for m in LEGACY_END.finditer(text) if start.end() < m.start() < end.start()]
    if legacy:
        if len(legacy) != 1 or legacy[0].start() < start.end() + .8 * (end.start() - start.end()):
            raise CorpusError("Ambiguous embedded Gutenberg end marker; review this edition.")
        end = legacy[0]
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
    r"(?im)^[ \t]*_?(?:CHAPTER|CHAP\.|LETTER|PART|BOOK)[ \t]+"
    r"(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|FIRST|SECOND|THIRD)(?=[\W_]|$)[^\n]{0,140}$"
)
PROLOGUE = re.compile(r"(?im)^[ \t]*(?:PROLOGUE|PRELUDE|INTRODUCTORY CHAPTER)[. \t]*$")
BACK_MATTER = re.compile(
    r"(?im)^[ \t]*(?:(?:A )?NOTE ON THE TEXT|TRANSCRIBER[’']?S? NOTES?|"
    r"TYPOGRAPHICAL ERRORS[^\n]*|FOOTNOTES|ENDNOTES|"
    r"PRINTED BY[^\n]*|(?-i:[A-Z .&'-]+PRESS:)[^\n]*|[^\n]{0,40}: PRESS OF[^\n]*)[.: \t]*$"
)
STORY_END = re.compile(r"(?im)^[ \t]*(?:THE END|FINIS|END OF VOL(?:UME)?\.?\s+[IVX\d]+)[.! \t]*$")

SUSPECT = {
    "gutenberg_boilerplate": r"(?i)project\s+gutenberg|gutenberg\.org|www\.pgdp\.net|distributed\s+proofread",
    "transcriber_material": r"(?im)\btranscrib(?:er|ers|er's|ers’|er’s).{0,15}(?:note|comment)|^[ \t]*(?:digitiz(?:ed|ation)\b|scann?ed\s+(?:by|from)\b)",
    "contents_or_illustration_list": r"(?im)^\s*(?:(?:table of )?contents|list of illustrations|illustrations)[. \t]*$",
    "publisher_material": r"(?im)^\s*(?:all rights reserved|printed (?:by|in)|published by|copyright(?:ed)?\b|advertisements\b)",
    "possible_back_matter": r"(?im)^\s*(?:by the same author|other books by|books by|publisher['’]s (?:note|catalogue|catalog|announcement)|appendix|editor['’]s notes?|(?:a )?note on the text|for sale by all booksellers|.*cloth,?\s+12mo).*",
    "remaining_illustration": r"(?i)\[illustration\b|\[page\s+\d+\]|\[pg\.?\s*\d+\]",
    "possible_footnote": r"(?m)\[(?:\d{1,4}|[A-Z]|[*†‡])\]|(?i:^\s*(?:footnotes?|endnotes?)\b)",
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
        headings = [*CHAPTER.finditer(text), *PROLOGUE.finditer(text)]
        bare = list(re.finditer(r"(?m)^[ \t]*(?:[IVXLCDM]{1,8})[.)]?[ \t]*$", text))
        if {"I", "II"}.issubset({m.group().strip().rstrip(".)") for m in bare}):
            headings.extend(bare)
        headings.sort(key=lambda m: m.start())
        candidates = []
        for index, heading in enumerate(headings):
            next_pos = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            after = text[heading.end():min(next_pos, heading.end() + 8000)]
            paragraphs = re.split(r"\n[ \t]*\n", after.strip())[:20]
            # Contents entries sit on adjacent lines; actual headings have a blank line.
            separated = bool(re.match(r"[ \t]*\n[ \t]*\n", text[heading.end():]) or
                             re.search(r"\n[ \t]*\n[ \t]*$", text[max(0, heading.start()-30):heading.start()]))
            prose_words = sum(len(p.split()) for p in paragraphs
                              if re.search(r"[.!?][\"’”']?(?:\s|$)", p) and len(p.split()) >= 4)
            # A final TOC entry can be followed by a long introduction before Chapter I.
            number = re.search(r"(?i)\b(?:CHAPTER|CHAP\.|LETTER|PART|BOOK)\s+([\w]+)", heading.group())
            label = number[1].upper() if number else heading.group().strip().rstrip(".)")
            restarts = label not in {"I", "1", "ONE", "FIRST", "PROLOGUE", "PRELUDE", "INTRODUCTORY CHAPTER"} and any(
                re.match(r"(?i)\s*(?:(?:CHAPTER|CHAP\.|LETTER|PART|BOOK)\s+)?(?:I|1|ONE|FIRST)\b", h.group())
                for h in headings[index + 1:index + 3])
            if separated and prose_words >= 30 and not restarts:
                candidates.append(heading)
        if candidates:
            start = candidates[0].start()
            method = "narrative_heading"
            if start > len(text) * 0.25:
                problems.append("large_front_matter_cut")
            # Do not silently drop an opening with too little prose for the heuristic.
            earlier = [h for h in headings if h.start() < start
                       and re.match(r"[ \t]*\n[ \t]*\n", text[h.end():])]
            if earlier:
                previous = earlier[-1]
                between = text[previous.end():start]
                if re.search(r'["“][^\n]+[.!?]', between) and not re.search(r"(?im)^\s*CONTENTS\b", between):
                    problems.append("possible_narrative_before_first_chapter")
        else:
            start = 0
            method = "unresolved"
            problems.append("narrative_start_not_identified")
    end = anchor_position(text, rules["end"]) if "end" in rules else len(text)
    if "end" not in rules:
        endings = []
        for marker in STORY_END.finditer(text):
            previous = text[:marker.start()].rstrip().rsplit("\n\n", 1)[-1].strip()
            # "THE END" can be the title of a final chapter, not the end of its prose.
            chapter_title = CHAPTER.fullmatch(previous) or re.fullmatch(r"[IVXLCDM]{1,8}[.)]?", previous)
            if marker.start() > max(start, len(text) * .65) and not chapter_title:
                endings.append(marker)
        if endings and not re.search(r"(?im)^\s*(?:EPILOGUE|CHAPTER|CHAP\.|LETTER|PART|BOOK)\b", text[endings[-1].end():]):
            end = endings[-1].end()
        editorial = [m for m in BACK_MATTER.finditer(text) if m.start() > max(start, len(text) * .65)]
        if editorial:
            end = min(end, editorial[0].start())
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


def strip_annotations(text, audit, footnotes=False):
    pattern = r"(?i)\[(?:Footnote\b" if footnotes else r"(?i)\[(?:Illustrations?\b|Transcriber[’']?s?\s+Notes?\b"
    pattern += ")"
    pieces, cursor = [], 0
    for match in re.finditer(pattern, text):
        if match.start() < cursor:
            continue
        depth, end = 0, None
        for position in range(match.start(), min(len(text), match.start() + 30000)):
            depth += (text[position] == "[") - (text[position] == "]")
            if depth == 0:
                end = position + 1
                break
        if end is None:
            raise CorpusError("Unclosed or oversized editorial annotation; add an explicit rule.")
        removed = text[match.start():end]
        # A chapter title may itself be typeset as an illustration. Retain that label.
        inner = re.sub(r"(?i)^\[Illustrations?\s*:?", "", removed).rstrip("]").strip()
        labels = [] if footnotes else [h.group().strip() for h in CHAPTER.finditer(inner)]
        replacement = "\n\n" + "\n\n".join(labels) + "\n\n" if labels else ""
        pieces.extend((text[cursor:match.start()], replacement))
        record_removal(audit, "labelled_footnote" if footnotes else "editorial_annotation", removed)
        cursor = end
    return "".join(pieces) + text[cursor:]


def remove_linked_footnotes(text, audit):
    # Remove a single-paragraph numbered note only when its inline reference exists.
    definitions = list(re.finditer(r"(?m)^[ \t]*\[(\d{1,4})\][ \t]+[^\n]+(?:\n(?![ \t]*\n)[^\n]+)*", text))
    for match in reversed(definitions):
        marker = f"[{match[1]}]"
        following = text[match.end():].lstrip()
        bounded = not following or CHAPTER.match(following) or re.match(r"\[\d{1,4}\]\s", following)
        if bounded and text[:match.start()].count(marker) == 1:
            record_removal(audit, "linked_footnote", match.group())
            text = text[:match.start()] + text[match.end():]
            text = text.replace(marker, "", 1)
    return text


def clean_book(data: bytes, rules: dict, keep_book_matter: bool = False, reflow: bool = True) -> tuple[str, dict]:
    if not isinstance(rules, dict) or set(rules) - {"raw_sha256", "start", "end", "remove", "accept_findings", "notes"}:
        raise CorpusError("Unknown cleaning-rule field; check the rule dictionary.")
    decoded, encoding = decode_book(data)
    body, metadata = gutenberg_body(decoded)
    audit, problems = [], []
    if rules and rules.get("raw_sha256") != sha256(data):
        raise CorpusError("Cleaning rules require the matching raw_sha256; the edition may have changed.")
    original_chars = len(body)
    # Explicit digital artifacts can occur within either boundary.
    note_ids = set(re.findall(r"(?i)\[Footnote\s+(\d{1,4}|[A-Z]|[*†‡])\s*:", body))
    note_sections = [m for m in re.finditer(r"(?im)^\s*(?:FOOTNOTES|ENDNOTES)[.: \t]*$", body)
                     if m.start() > len(body) * .65]
    if note_sections:
        note_ids.update(re.findall(r"(?m)^[ \t]*\[(\d{1,4}|[A-Z]|[*†‡])\]", body[note_sections[-1].end():]))
    body = strip_annotations(body, audit)
    # A terminal transcriber's section is safely bounded only near the end.
    notes = list(re.finditer(r"(?im)^(?:[ \t]*\+[-=+]{3,}\+[ \t]*\n(?:[ \t]*\n)*)?[ \t*_=#●•|+-]*TRANSCRIBER[’']?S?[ \t]+NOTES?(?:[ \t*_=#|]*[.:][^\n]*|[ \t*_=#|]*)$", body))
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
        # Remove labelled or clearly paired notes; flag ambiguous references for review.
        body = strip_annotations(body, audit, footnotes=True)
        body = remove_linked_footnotes(body, audit)
        body = substitute(body, r"(?im)^[ \t]*FOOTNOTES?[:. \t]*\n(?:[ \t]*\n)+(?=[ \t]*(?:CHAPTER|CHAP\.|LETTER|PART|BOOK)\b)",
                          "empty_footnote_heading", audit)
        if note_ids:
            reference_pattern = r"\[(?:" + "|".join(re.escape(note) for note in sorted(note_ids)) + r")\]"
            body = substitute(body, reference_pattern, "defined_footnote_reference", audit)
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
    for url in entries:
        if not isinstance(url, str) or not url.strip():
            raise CorpusError("Each corpus entry must be a nonempty Gutenberg URL string.")
        book = Book(gutenberg_id(url), url)
        books.setdefault(book.id, book)
    if not books:
        raise CorpusError("The corpus list is empty.")
    return list(books.values())


def overlapping_editions(documents):
    """Flag substantial shared prose across splits without deleting narrative passages."""
    index, totals, overlaps = {}, {}, {}
    for doc in documents:
        paragraphs = {}
        for paragraph in doc["text"].split("\n\n"):
            words = re.findall(r"\w+", unicodedata.normalize("NFKC", paragraph).casefold())
            if len(words) >= 40:
                paragraphs[sha256(" ".join(words).encode())] = len(words)
        totals[doc["id"]] = sum(paragraphs.values())
        shared = {}
        for digest, words in paragraphs.items():
            for previous, split in index.get(digest, []):
                if split != doc["split"]:
                    count, length = shared.get(previous, (0, 0))
                    shared[previous] = count + 1, length + words
        hits = [previous for previous, (count, length) in shared.items()
                if count >= 3 and length >= .2 * min(totals[previous], totals[doc["id"]])]
        if hits:
            overlaps[doc["id"]] = hits
        else:
            for digest in paragraphs:
                index.setdefault(digest, []).append((doc["id"], doc["split"]))
    return overlaps


def prepare_books(entries, rules, output, allow_partial=False, offline=False):
    downloader = Downloader(output, DEFAULT_MIRROR, 2, 30, 2, offline, cache_raw=False)
    documents, audit, seen = [], [], set()
    for book in load_books(entries):
        row = {"id": book.id, "url": book.url}
        try:
            raw, metadata = downloader.fetch(book)
            row["raw_sha256"] = metadata["sha256"]
            text, report = clean_book(raw, rules.get(book.id, {}))
            digest = sha256(" ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())).encode())
            row.update(report, raw_sha256=metadata["sha256"], opening=text[:500], ending=text[-500:])
            row["status"] = "needs_review" if report["problems"] else "duplicate" if digest in seen else "accepted"
            if row["status"] == "accepted":
                seen.add(digest)
                documents.append({"id": book.id, "text": text,
                                  "split": split_for(book.id, "gutenberg-corpus-v1", .05, .05)})
        except (CorpusError, OSError, ValueError) as exc:
            row.update(status="failed", error=str(exc))
        audit.append(row)
        print(book.id, row["status"], row.get("problems", row.get("error", "")))
    overlaps = overlapping_editions(documents)
    for row in audit:
        if row["id"] in overlaps:
            row.update(status="needs_review", problems=["shared_prose_across_splits"], overlaps_with=overlaps[row["id"]])
            print(row["id"], "shares substantial prose with", row["overlaps_with"], "— review the overlap and keep one edition if duplicated.")
    documents = [doc for doc in documents if doc["id"] not in overlaps]
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
    """Book-local windows; epoch offsets vary context without losing prediction targets."""
    def __init__(self, texts, tokenizer, context_length):
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        self.context_length = context_length
        self.text_bytes = sum(len(text.encode("utf-8")) for text in texts)
        self.books = [np.asarray([tokenizer.bos_token_id,
                      *tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id],
                      dtype=np.int32) for text in texts]
        self.first_targets = [context_length - 1] * len(self.books)
        self._layout()

    def _layout(self):
        counts = [1 + math.ceil(max(0, len(book) - 1 - first) / (self.context_length - 1))
                  for book, first in zip(self.books, self.first_targets)]
        self.ends = np.cumsum(counts).tolist()

    def set_epoch(self, seed, epoch):
        # Independent RNG: the same epoch has identical windows when resuming.
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        self.first_targets = rng.integers(1, self.context_length, size=len(self.books)).tolist()
        self._layout()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        book_index = bisect_right(self.ends, index)
        local = index - (self.ends[book_index - 1] if book_index else 0)
        first = self.first_targets[book_index]
        start = 0 if local == 0 else first + (local - 1) * (self.context_length - 1)
        end = first + 1 if local == 0 else start + self.context_length
        return self.books[book_index][start:end].tolist()


def collate_windows(windows, pad_token_id, target_starts=None):
    ids = torch.full((len(windows), max(map(len, windows))), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(windows):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    if target_starts is not None:
        for row, start in enumerate(target_starts):
            labels[row, :start] = -100
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


def evaluation_windows(dataset, stride):
    """Give held-out continuations context; count every target exactly once."""
    stride = min(stride, dataset.context_length - 1)
    for book in dataset.books:
        previous_end = 1  # BOS is context, never a prediction target.
        for start in range(0, len(book) - 1, stride):
            end = min(start + dataset.context_length, len(book))
            yield book[start:end].tolist(), max(1, previous_end - start)
            previous_end = end
            if end == len(book):
                break


def evaluate(model, dataset, config, tokenizer, device):
    if not len(dataset):
        return None
    model.eval()
    total_nll, targets = 0.0, 0
    windows, starts = [], []
    def score():
        batch = collate_windows(windows, tokenizer.pad_token_id, starts)
        count = int((batch["labels"][:, 1:] != -100).sum())
        batch = {key: value.to(device) for key, value in batch.items()}
        with amp_context(device):
            loss = float(model(**batch).loss)
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss.")
        return loss * count, count
    with torch.inference_mode():
        for tokens, target_start in evaluation_windows(dataset, config.eval_stride):
            windows.append(tokens)
            starts.append(target_start)
            if len(windows) == config.batch_size:
                nll, count = score()
                total_nll += nll
                targets += count
                windows, starts = [], []
        if windows:
            nll, count = score()
            total_nll += nll
            targets += count
    loss = total_nll / targets
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 700 else None,
            "target_tokens": targets, "bits_per_byte": total_nll / (math.log(2) * dataset.text_bytes) if dataset.text_bytes else None}


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

def train(datasets, tokenizer, config, output, resume=False, max_seconds=0, device_name="cuda"):
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
    recipe = {"pipeline": "v3-epoch-offsets-sliding-eval", "config": settings, "data": data_hash.hexdigest(),
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
    steps_per_epoch = []
    for epoch in range(config.epochs):
        datasets["train"].set_epoch(config.seed, epoch)
        batches = math.ceil(len(datasets["train"]) / config.batch_size)
        steps_per_epoch.append(math.ceil(batches / config.gradient_accumulation))
    datasets["train"].set_epoch(config.seed, 0)
    total_steps = sum(steps_per_epoch)
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
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    state = {"epoch": 0, "next_batch": 0, "step": 0, "tokens_seen": 0, "best_loss": None, "best_step": None, "best_epoch_loss": None, "bad_epochs": 0}
    checkpoints = checkpoint_paths(output)
    if resume and checkpoints:
        state = restore_checkpoint(checkpoints[-1], model, optimizer, scheduler, identity, device)
        print(f"Resuming optimizer step {state['step']}.", flush=True)
    elif resume:
        print("No complete checkpoint yet; restarting this initialized run from its seed.", flush=True)
        # Tokenizer training may use its own RNG; model initialization uses the seed above.
    summary = {"parameters": sum(p.numel() for p in model.parameters()), "vocab_size": len(tokenizer),
               "context_length": config.context_length, "maximum_epochs": config.epochs,
               "splits": {s: {"books": len(datasets[s].books), "target_tokens": sum(len(book) - 1 for book in datasets[s].books), "windows": len(datasets[s])} for s in datasets},
               "planned_optimizer_steps": total_steps, "steps_per_epoch": steps_per_epoch,
               "maximum_tokens_per_update": config.batch_size * config.gradient_accumulation * (config.context_length - 1), "hardware": hardware}
    write_json(output / "training_plan.json", summary)
    print(json.dumps(summary, indent=2), flush=True)

    started, session_tokens = time.monotonic(), 0
    early_stopped = bool(config.early_stopping_patience and state["bad_epochs"] >= config.early_stopping_patience)
    stop_reason = "early_stopping" if early_stopped else "complete"
    while not early_stopped and state["step"] < total_steps and state["epoch"] < config.epochs:
        epoch = state["epoch"]
        datasets["train"].set_epoch(config.seed, epoch)
        batches_per_epoch = math.ceil(len(datasets["train"]) / config.batch_size)
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
                if batch_end == batches_per_epoch:
                    if state["best_epoch_loss"] is None or metrics["loss"] < state["best_epoch_loss"] - config.early_stopping_min_delta:
                        state["best_epoch_loss"], state["bad_epochs"] = metrics["loss"], 0
                    else:
                        state["bad_epochs"] += 1
                    early_stopped = bool(config.early_stopping_patience and state["bad_epochs"] >= config.early_stopping_patience)
            if should_eval or state["step"] % config.save_every == 0 or finished or timed_out:
                save_checkpoint(output, model, optimizer, scheduler, state, identity, config)
            if finished or timed_out or early_stopped:
                stop_reason = "early_stopping" if early_stopped else "complete" if finished else "session_time_limit"
                break
        if stop_reason in {"session_time_limit", "early_stopping"}:
            break

    # Export the validation-selected model; test only at completion or early stopping.
    from safetensors.torch import load_model
    best = output / f"checkpoint-{state['best_step']}"
    load_model(model, str(best / "model.safetensors"), device=str(device))
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.gradient_checkpointing_disable()
    model.save_pretrained(output / "best_model", safe_serialization=True)
    tokenizer.model_max_length = config.context_length
    tokenizer.save_pretrained(output / "best_model")
    complete = early_stopped or state["step"] >= total_steps or state["epoch"] >= config.epochs
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
