"""
Cell guide:
  1. Google Drive, paths, session settings
  2. Editable corpus list (append books here)
  3. Editable cleaning rules
  4. Editable model/training settings and 100-step sample prompts
  5. Dependencies and GPU check
  6. Shared corpus registry
  7. Corpus download, cleaning and audit code
  8. Tokenizer and training-data code
  9. GPT-2 training code, checkpoints and progress samples
 10. Generation code
 11. Prepare and review the corpus
 12. Train or resume (samples print every 100 optimizer updates)
 13. Generate from the saved model

Cells 2–10 contain the existing files verbatim inside raw triple-quoted strings.
Run each whole cell: it writes that source into the project directory on Drive.
Keep the surrounding write_text call and quotes. This preserves the existing
imports, __file__ paths, command-line entry points and checkpoint fingerprints;
no cleaning, tokenization, training, sampling or generation logic was rewritten.
You never need to create or upload those modules yourself.

For this copy/paste workflow, expand the list in Cell 2 and change configuration
in Cells 3–4. Each run of those cells writes their displayed contents to Drive.
If you previously edited the Drive copies directly, carry those edits into the
corresponding cells first. After corpus/model changes, choose a new RUN_NAME in
Cell 1 and set REBUILD_CORPUS=True when cleaning inputs changed. sample_* preview
settings can change on resume. Preserve the rest of the source in Cells 6–10.

To resume in a fresh Colab runtime, rerun the cells with the same RUN_NAME and
training inputs. Cell 12 detects the existing run and resumes automatically.
This file is the sole application source; its cells create all required modules.
"""

# ============================================================================
# CELL 1: GOOGLE DRIVE, PATHS AND SESSION SETTINGS
# ============================================================================

from google.colab import drive
from pathlib import Path
import json
import os
import subprocess
import sys

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/GPT2Corpus')
PROJECT = ROOT / 'project'
CORPUS = ROOT / 'corpus'
RUN_NAME = 'gpt2-001'  # Choose a new name for new data, architecture, or schedule.
OUTPUT = ROOT / 'runs' / RUN_NAME
MAX_SESSION_SECONDS = 2 * 60 * 60
ALLOW_PARTIAL_CORPUS = False  # True trains accepted books only, after audit review.
REBUILD_CORPUS = False       # True re-cleans cached books and downloads new entries.
PROJECT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
print('Project:', PROJECT)
print('Training output:', OUTPUT)

# ============================================================================
# CELL 2: CORPUS LIST — APPEND BOOKS HERE
# ============================================================================

# Edit the sources list below; there is no fixed number of books.
# BEGIN UNCHANGED FILE: corpus-list.py
(PROJECT / "corpus-list.py").write_text(r'''# Single source of truth: append URLs freely; no fixed length is used.
# For stable grouping across volumes/editions, entries may also be dictionaries:
# {"url": "https://www.gutenberg.org/ebooks/123", "work_id": "author-work", "note": "Title by Author"}
# The isolated corpus list along with notes on authors, fecundity, genre, time period, etc.
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
]
''', encoding="utf-8")
# END UNCHANGED FILE: corpus-list.py
print("Ready: corpus-list.py")

# ============================================================================
# CELL 3: CLEANING RULES — EDIT HERE WHEN REVIEWING AN EDITION
# ============================================================================

# Keep this JSON valid; add checksum-bound rules for flagged editions here.
# BEGIN UNCHANGED FILE: cleaning_rules.json
(PROJECT / "cleaning_rules.json").write_text(r'''{
  "version": 1,
  "books": {}
}
''', encoding="utf-8")
# END UNCHANGED FILE: cleaning_rules.json
print("Ready: cleaning_rules.json")

# ============================================================================
# CELL 4: MODEL, TRAINING AND PROGRESS-SAMPLE SETTINGS — EDIT HERE
# ============================================================================

# Defaults: approximately 111M parameters; sample_every=100 optimizer updates.
# BEGIN UNCHANGED FILE: training_config.json
(PROJECT / "training_config.json").write_text(r'''{
  "vocab_size": 32768,
  "min_frequency": 2,
  "context_length": 1024,
  "n_layer": 12,
  "n_head": 12,
  "n_embd": 768,
  "dropout": 0.1,
  "batch_size": 8,
  "gradient_accumulation": 8,
  "epochs": 3,
  "max_steps": 0,
  "learning_rate": 0.0003,
  "weight_decay": 0.1,
  "warmup_fraction": 0.03,
  "max_grad_norm": 1.0,
  "eval_every": 200,
  "save_every": 100,
  "log_every": 10,
  "keep_checkpoints": 2,
  "gradient_checkpointing": true,
  "seed": 42,
  "sample_every": 100,
  "sample_max_new_tokens": 128,
  "sample_temperature": 0.8,
  "sample_top_p": 0.95,
  "sample_seed": 1234,
  "sample_prompts": [
    "The evening was quiet, and",
    "She opened the letter and",
    "\"I never expected,\" she said,"
  ]
}
''', encoding="utf-8")
# END UNCHANGED FILE: training_config.json
print("Ready: training_config.json")

# ============================================================================
# CELL 5: DEPENDENCIES AND A100 CHECK
# ============================================================================

# The same dependency versions as the existing workflow.
# BEGIN UNCHANGED FILE: requirements-colab.txt
(PROJECT / "requirements-colab.txt").write_text(r'''# Use Colab's CUDA-enabled PyTorch; do not replace it with a CPU-only wheel.
# These Transformers / Tokenizers versions are the tested API pair.
torch>=2.6,<3
transformers==5.2.0
tokenizers==0.22.2
numpy>=1.26,<3
safetensors>=0.4.3,<1
''', encoding="utf-8")
# END UNCHANGED FILE: requirements-colab.txt
print("Ready: requirements-colab.txt")

# Install before importing ML packages. Keep Colab's CUDA-enabled torch.
# If Colab asks for a runtime restart, restart and rerun from the first cell.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PROJECT / 'requirements-colab.txt')], check=True)
subprocess.run(['nvidia-smi'], check=True)

# ============================================================================
# CELL 6: SHARED CORPUS REGISTRY — EXISTING CODE
# ============================================================================

# Run this cell to install the unchanged registry module.
# BEGIN UNCHANGED FILE: corpus_registry.py
(PROJECT / "corpus_registry.py").write_text(r'''"""Shared, data-only corpus registry; the editable source stays in corpus-list.py."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import tokenize
import unicodedata
from urllib.parse import urlparse

DEFAULT_SOURCES = Path(__file__).with_name("corpus-list.py")


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


def load_books(path: Path = DEFAULT_SOURCES) -> list[Book]:
    """Parse literals, never execute source code. No corpus-size/title assumptions.

    Entries can be URL strings with comments, or dictionaries containing url,
    note, and a stable work_id shared by related volumes/editions.
    """
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    values = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "sources" for t in targets):
            values.append(node.value)
    if len(values) != 1 or not isinstance(values[0], ast.List):
        raise CorpusError("Expected exactly one literal Python list named sources.")
    comments = {t.start[0]: t.string.lstrip("# ").strip() for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT}
    books, seen = [], {}
    for element in values[0].elts:
        try:
            value = ast.literal_eval(element)
        except (ValueError, TypeError) as exc:
            raise CorpusError("Every sources entry must be a literal URL or dictionary.") from exc
        if isinstance(value, str):
            value = {"url": value}
        if not isinstance(value, dict) or set(value) - {"url", "note", "work_id"} or not isinstance(value.get("url"), str):
            raise CorpusError("Corpus dictionaries support only url, note, and work_id.")
        if any(not isinstance(v, str) or not v.strip() for v in value.values()):
            raise CorpusError("Corpus entry fields must be nonempty strings.")
        book = Book(gutenberg_id(value["url"]), value["url"], value.get("note", comments.get(element.end_lineno, "")), value.get("work_id", ""))
        if book.id in seen:
            if seen[book.id].work_id != book.work_id:
                raise CorpusError(f"Conflicting work_id values for duplicate ebook {book.id}.")
            continue
        books.append(book)
        seen[book.id] = book
    if not books:
        raise CorpusError("The sources list is empty.")
    return books


def work_key(book: Book, rules: dict) -> str:
    if rules.get("work_id"):
        return rules["work_id"]
    if book.work_id:
        return book.work_id
    # Existing comments are supported; explicit work_id is safer for new entries.
    title = re.split(r"\s+-\s+", book.note, maxsplit=1)[0]
    title = re.sub(r"(?i)\(?\b(?:vol(?:ume)?s?\.?)\s*(?:\d+|[ivx]+)(?:\s*(?:of|[-–])\s*(?:\d+|[ivx]+))?\)?", "", title)
    title = re.sub(r"(?i)\(of\s+\d+\)", "", title)
    title = re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC", title).casefold()).strip()
    return title or f"gutenberg-{book.id}"


def split_for(work_id: str, seed: str, validation: float, test: float) -> str:
    value = int(hashlib.sha256(f"{seed}\0{work_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value < test else "validation" if value < test + validation else "train"
''', encoding="utf-8")
# END UNCHANGED FILE: corpus_registry.py
print("Ready: corpus_registry.py")

# ============================================================================
# CELL 7: CORPUS DOWNLOAD, CLEANING AND AUDIT — EXISTING CODE
# ============================================================================

# Run this cell to install the unchanged corpus-preparation module.
# BEGIN UNCHANGED FILE: prepare_corpus.py
(PROJECT / "prepare_corpus.py").write_text(r'''#!/usr/bin/env python3
"""Download, audit, and prepare a Gutenberg corpus. Python 3.10+, no dependencies.

Trying to be extra careful with the parser since this is a from-scratch model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime

from corpus_registry import Book, CorpusError, DEFAULT_SOURCES, load_books, split_for, work_key

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


# Needs definite enhancement
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


def chunks(text: str, max_words: int):
    """Nonoverlapping character slices. Concatenating chunks restores exact text."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    start, count = 0, 0
    for word in re.finditer(r"\S+", text):
        if count == max_words:
            yield text[start:word.start()]
            start, count = word.start(), 0
        count += 1
    if text[start:].strip():
        yield text[start:]


def load_rules(path: Path) -> dict:
    if not path.exists():
        raise CorpusError(f"Cleaning rules file not found: {path}")
    rules = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or rules.get("version") != 1 or not isinstance(rules.get("books"), dict):
        raise CorpusError('Rules must contain "version": 1 and a "books" object.')
    allowed = {"raw_sha256", "start", "end", "remove", "accept_findings", "work_id", "notes"}
    for book_id, rule in rules["books"].items():
        if not book_id.isdigit() or not isinstance(rule, dict) or set(rule) - allowed:
            raise CorpusError(f"Invalid or unknown rule fields for {book_id}.")
        if "work_id" in rule and (not isinstance(rule["work_id"], str) or not rule["work_id"].strip()):
            raise CorpusError(f"work_id must be a nonempty string for {book_id}.")
    return rules["books"]


def prepare(books: list[Book], downloader: Downloader, args) -> int:
    rules = load_rules(args.overrides)
    # A new directory per invocation prevents failed/partial runs reusing stale text.
    run = args.output / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True)
    # Preserve the exact editable inputs with the immutable output run.
    (run / "corpus-list.py").write_bytes(args.sources.read_bytes())
    (run / "cleaning_rules.json").write_bytes(args.overrides.read_bytes())
    records, seen_hashes, counts = [], {}, Counter()
    outputs = {}
    try:
        for split in ("train", "validation", "test"):
            outputs[split] = (run / f"{split}.jsonl").open("w", encoding="utf-8")
        with (run / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for index, book in enumerate(books, 1):
                record = {"id": book.id, "source_url": book.url, "source_note": book.note}
                try:
                    data, source = downloader.fetch(book)
                    record["download"] = source
                    clean, details = clean_book(data, rules.get(book.id, {}), args.keep_book_matter, not args.preserve_linebreaks)
                    record.update(details)
                    record["clean_sha256"] = sha256(clean.encode("utf-8"))
                    record["opening"], record["closing"] = clean[:800], clean[-800:]
                    if details["problems"]:
                        record["status"] = "needs_review"
                        if args.write_books:
                            target = run / "review" / f"{book.id}.txt"
                            target.parent.mkdir(exist_ok=True)
                            target.write_text(clean, encoding="utf-8")
                    elif record["clean_sha256"] in seen_hashes:
                        record.update(status="duplicate", duplicate_of=seen_hashes[record["clean_sha256"]])
                    else:
                        record["status"] = "accepted"
                        seen_hashes[record["clean_sha256"]] = book.id
                        work_id = work_key(book, rules.get(book.id, {}))
                        split = split_for(work_id, args.seed, args.validation_fraction, args.test_fraction)
                        record.update(work_id=work_id, split=split, samples=0)
                        for number, chunk in enumerate(chunks(clean, args.chunk_words)):
                            sample = {"id": f"{book.id}:{number}", "book_id": book.id, "work_id": work_id, "text": chunk, "chunk_index": number}
                            outputs[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                            record["samples"] += 1
                        counts[f"{split}_books"] += 1
                        counts[f"{split}_samples"] += record["samples"]
                        counts["accepted_words"] += details["words"]
                        if args.write_books:
                            target = run / "clean" / f"{book.id}.txt"
                            target.parent.mkdir(exist_ok=True)
                            target.write_text(clean, encoding="utf-8")
                except (CorpusError, ValueError) as exc:
                    record.update(status="failed", error=str(exc))
                counts[record["status"]] += 1
                records.append(record)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                print(f"[{index}/{len(books)}] {book.id}: {record['status']}" + (f" — {record['error']}" if "error" in record else ""), flush=True)
    finally:
        for output in outputs.values():
            output.close()
    report = {"version": VERSION, "requested": len(books), "counts": dict(counts), "complete": not (counts["failed"] or counts["needs_review"]), "sources_sha256": sha256((run / "corpus-list.py").read_bytes()), "rules_sha256": sha256((run / "cleaning_rules.json").read_bytes()), "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(run / "report.json", report)
    write_audit(run / "audit.html", records, report)
    write_json(args.output / "latest.json", {"run": str(run.resolve()), "relative_run": str(run.relative_to(args.output)), "complete": report["complete"]})
    print(f"Outputs: {run}\nAccepted: {counts['accepted']}; review: {counts['needs_review']}; failed: {counts['failed']}; duplicates: {counts['duplicate']}", flush=True)
    return 0 if report["complete"] else 1


def write_audit(path: Path, records: list[dict], report: dict) -> None:
    escape = lambda value: html.escape(str(value))
    with path.open("w", encoding="utf-8") as out:
        out.write('<!doctype html><meta charset="utf-8"><title>Corpus cleaning audit</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:16px}article{border-top:1px solid #bbb;padding:20px 0}summary{cursor:pointer}</style><h1>Corpus cleaning audit</h1>')
        out.write("<p>Only accepted books appear in training JSONL. Automated checks do not certify editorial completeness. Review openings, endings, and flagged material.</p>")
        out.write(f"<pre>{escape(json.dumps(report['counts'], indent=2))}</pre>")
        for record in records:
            out.write(f"<article><h2>{escape(record['id'])}: {escape(record.get('source_note', ''))}</h2><p>Status: {escape(record['status'])}</p>")
            for name in ("error", "problems", "opening", "closing"):
                if record.get(name):
                    out.write(f"<h3>{escape(name.title())}</h3><pre>{escape(record[name])}</pre>")
            out.write(f"<details><summary>Provenance and removals</summary><pre>{escape(json.dumps(record, ensure_ascii=False, indent=2))}</pre></details></article>")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=Path("data/corpus"))
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help="Gutenberg collection root (uses the official PGLAF mirror by default)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between requests (default: 2)")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+", help="Process selected ebook IDs")
    parser.add_argument("--offline", action="store_true", help="Use verified local downloads only")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--cache-raw", action="store_true", help="Keep original downloads on disk (off by default; implied by --download-only)")
    parser.add_argument("--write-books", action="store_true", help="Also write one text file per accepted/review book (off by default to save space)")
    parser.add_argument("--preserve-linebreaks", action="store_true", help="Do not unwrap prose lines")
    parser.add_argument("--overrides", type=Path, default=Path(__file__).with_name("cleaning_rules.json"))
    parser.add_argument("--keep-book-matter", action="store_true", help="Retain original front/back matter and footnotes; still remove digital boilerplate")
    parser.add_argument("--chunk-words", type=int, default=1024, help="Maximum whitespace-delimited words per training sample; not a tokenizer count")
    parser.add_argument("--seed", default="gutenberg-corpus-v1")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or not math.isfinite(args.timeout) or args.delay < 0 or args.timeout <= 0 or args.retries < 0 or (args.limit is not None and args.limit <= 0) or args.chunk_words <= 0:
        parser.error("Delay/retries must be nonnegative; timeout, limit, and chunk-words must be positive.")
    if not (0 <= args.validation_fraction < 1 and 0 <= args.test_fraction < 1 and args.validation_fraction + args.test_fraction < 1):
        parser.error("Split fractions must be nonnegative and sum to less than 1.")
    if urlparse(args.mirror).scheme != "https":
        parser.error("--mirror must use HTTPS.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        books = load_books(args.sources)
        if args.ids:
            unknown = set(args.ids) - {b.id for b in books}
            if unknown:
                raise CorpusError(f"Unknown book IDs: {sorted(unknown)}")
            books = [b for b in books if b.id in args.ids]
        if args.limit:
            books = books[:args.limit]
        downloader = Downloader(args.output, args.mirror, args.delay, args.timeout, args.retries, args.offline, args.cache_raw or args.download_only)
        if args.download_only:
            failures = []
            for index, book in enumerate(books, 1):
                try:
                    data, _ = downloader.fetch(book)
                    print(f"[{index}/{len(books)}] {book.id}: {len(data):,} bytes", flush=True)
                except (CorpusError, OSError, ValueError) as exc:
                    failures.append({"id": book.id, "error": str(exc)})
                    print(f"[{index}/{len(books)}] {book.id}: ERROR {exc}", file=sys.stderr, flush=True)
            write_json(args.output / "download_report.json", {"requested": len(books), "downloaded": len(books) - len(failures), "failures": failures})
            return 1 if failures else 0
        return prepare(books, downloader, args)
    except (CorpusError, OSError, ValueError, SyntaxError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")
# END UNCHANGED FILE: prepare_corpus.py
print("Ready: prepare_corpus.py")

# ============================================================================
# CELL 8: TOKENIZER AND TRAINING DATA — EXISTING CODE
# ============================================================================

# Run this cell to install the unchanged tokenizer/data module.
# BEGIN UNCHANGED FILE: training_data.py
(PROJECT / "training_data.py").write_text(r'''"""Audited corpus -> train-only byte BPE -> lossless, book-bounded token windows."""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from corpus_registry import CorpusError
from prepare_corpus import sha256, write_json

SPLITS = ("train", "validation", "test")
DATA_VERSION = 1
SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run(path: Path) -> Path:
    if (path / "report.json").is_file():
        return path
    latest = path / "latest.json"
    if not latest.is_file():
        raise CorpusError(f"No prepared corpus at {path}; run prepare_corpus.py first.")
    pointer = json.loads(latest.read_text())
    # Relative paths survive copying a preparation run from a laptop to Colab.
    if pointer.get("relative_run"):
        target = path / pointer["relative_run"]
    else:
        target = path / "runs" / Path(pointer["run"]).name
        if not target.is_dir():
            target = Path(pointer["run"])
    if not (target / "report.json").is_file():
        raise CorpusError(f"Preparation run is missing: {target}")
    return target


def manifest_for(run: Path) -> dict:
    accepted = {}
    works, hashes = {}, set()
    with (run / "manifest.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["status"] != "accepted":
                continue
            book_id, work_id, split = row["id"], row["work_id"], row["split"]
            if split not in SPLITS or book_id in accepted:
                raise CorpusError("Invalid split or repeated accepted book in manifest.")
            if work_id in works and works[work_id] != split:
                raise CorpusError(f"Work {work_id!r} leaks across splits.")
            if row["clean_sha256"] in hashes:
                raise CorpusError("Duplicate accepted text in manifest.")
            works[work_id] = split
            hashes.add(row["clean_sha256"])
            accepted[book_id] = row
    return accepted


def iter_documents(run: Path, split: str, manifest: dict | None = None):
    """Reconstruct and verify one book at a time, including all final chunks."""
    manifest = manifest if manifest is not None else manifest_for(run)
    current, pieces, seen = None, [], set()

    def finish():
        if current is None:
            return None
        record = manifest[current]
        text = "".join(pieces)
        if len(pieces) != record["samples"] or sha256(text.encode("utf-8")) != record["clean_sha256"]:
            raise CorpusError(f"Missing/reordered/changed chunks for book {current}.")
        if not text.strip():
            raise CorpusError(f"Empty accepted book {current}.")
        return {"book_id": current, "work_id": record["work_id"], "text": text}

    with (run / f"{split}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            book_id = row["book_id"]
            if book_id not in manifest or manifest[book_id]["split"] != split or row["work_id"] != manifest[book_id]["work_id"]:
                raise CorpusError(f"Unaccepted book or wrong split/work in {split}: {book_id}")
            if book_id != current:
                document = finish()
                if document:
                    yield document
                if book_id in seen:
                    raise CorpusError(f"Noncontiguous chunks for {book_id}.")
                seen.add(book_id)
                current, pieces = book_id, []
            if row["id"] != f"{book_id}:{len(pieces)}" or row.get("chunk_index", len(pieces)) != len(pieces):
                raise CorpusError(f"Out-of-order chunk for {book_id}.")
            if not isinstance(row["text"], str):
                raise CorpusError("Training text must be a string.")
            pieces.append(row["text"])
    document = finish()
    if document:
        yield document
    expected = {book_id for book_id, row in manifest.items() if row["split"] == split}
    if seen != expected:
        raise CorpusError(f"Missing accepted books in {split}: {sorted(expected - seen)}")


def inspect_corpus(run: Path, allow_partial: bool = False) -> dict:
    report = json.loads((run / "report.json").read_text())
    if not report["complete"] and not allow_partial:
        raise CorpusError("Corpus has review items/download failures. Inspect audit.html; use --allow-partial-corpus to explicitly train only accepted books.")
    manifest = manifest_for(run)
    counts = {}
    for split in SPLITS:
        count, chars = 0, 0
        for document in iter_documents(run, split, manifest):
            count += 1
            chars += len(document["text"])
        counts[split] = {"books": count, "characters": chars}
    if not counts["train"]["books"] or not counts["validation"]["books"]:
        raise CorpusError("Training needs nonempty train and validation splits. Add works or choose a different preparation --seed / split fraction; never split chunks of the same work across sets.")
    hashes = {name: file_sha256(run / name) for name in ("manifest.jsonl", *(f"{s}.jsonl" for s in SPLITS))}
    fingerprint = sha256(json.dumps(hashes, sort_keys=True).encode())
    return {"fingerprint": fingerprint, "files": hashes, "counts": counts, "complete": report["complete"]}


def train_tokenizer(run: Path, destination: Path, vocab_size: int, min_frequency: int):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    backend = ByteLevelBPETokenizer(add_prefix_space=False)
    # Never expose validation/test texts to tokenizer training.
    backend.train_from_iterator(
        (doc["text"] for doc in iter_documents(run, "train")),
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS), show_progress=True,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend._tokenizer, bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[0], unk_token=None, pad_token=SPECIAL_TOKENS[1],
        # Full books are encoded before windowing, so don't truncate to model size.
        model_max_length=10**12, padding_side="right",
    )
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination)
    return tokenizer


def tokenize_corpus(run: Path, destination: Path, tokenizer) -> dict:
    import numpy as np

    destination.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in SPLITS:
        books, offset = [], 0
        token_path = destination / f"{split}.bin"
        with token_path.open("wb") as out:
            for doc in iter_documents(run, split):
                # Literal special-token strings in a source must not turn into controls.
                if any(token in doc["text"] for token in SPECIAL_TOKENS):
                    raise CorpusError(f"Reserved tokenizer control token in book {doc['book_id']}.")
                ids = tokenizer.encode(doc["text"], add_special_tokens=False, truncation=False)
                ids = [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
                np.asarray(ids, dtype="<u4").tofile(out)
                books.append({"book_id": doc["book_id"], "offset": offset, "length": len(ids)})
                offset += len(ids)
        stats[split] = {"tokens": offset, "target_tokens": offset - len(books), "books": books, "sha256": file_sha256(token_path)}
    write_json(destination / "index.json", stats)
    return stats


class TokenWindows:
    """Memory-mapped windows that never cross books and lose no target tokens.

    A single context token overlaps neighboring windows. BOS predicts the first
    text token; the final text token predicts EOS. Padding is not a target.
    """
    def __init__(self, directory: Path, split: str, context_length: int):
        import numpy as np
        if context_length < 2:
            raise ValueError("context_length must be at least two")
        self.context_length = context_length
        info = json.loads((directory / "index.json").read_text())[split]
        self.books = info["books"]
        self.tokens = np.memmap(directory / f"{split}.bin", mode="r", dtype="<u4") if info["tokens"] else []
        self.ends, count = [], 0
        for book in self.books:
            count += math.ceil((book["length"] - 1) / (context_length - 1))
            self.ends.append(count)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        book_index = bisect_right(self.ends, index)
        book = self.books[book_index]
        local = index - (self.ends[book_index - 1] if book_index else 0)
        start = book["offset"] + local * (self.context_length - 1)
        end = min(start + self.context_length, book["offset"] + book["length"])
        return self.tokens[start:end].astype("int64").tolist()


def collate_windows(windows, pad_token_id: int):
    import torch
    width = max(map(len, windows))
    ids = torch.full((len(windows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(windows):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    # EOS is a real target: pad has its own ID and is masked by attention mask.
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
''', encoding="utf-8")
# END UNCHANGED FILE: training_data.py
print("Ready: training_data.py")

# ============================================================================
# CELL 9: GPT-2 TRAINING, CHECKPOINTS AND 100-STEP SAMPLES — EXISTING CODE
# ============================================================================

# Run this cell to install the unchanged trainer; training starts in Cell 12.
# BEGIN UNCHANGED FILE: train_gpt2.py
(PROJECT / "train_gpt2.py").write_text(r'''#!/usr/bin/env python3
"""Train a randomly initialized GPT2LMHeadModel on the audited corpus (single GPU)."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

from corpus_registry import CorpusError, DEFAULT_SOURCES
import prepare_corpus
from prepare_corpus import sha256, write_json
from training_data import (DATA_VERSION, SPLITS, TokenWindows, collate_windows,
                           file_sha256, inspect_corpus, resolve_run,
                           tokenize_corpus, train_tokenizer)


# The preceding trainer differs only in progress sampling and its configuration.
# Allow its checkpoints after checking every training/data/environment setting.
PRE_SAMPLE_TRAINER_SHA256 = "e6bbb7324298a7187b8b9b5d605a642f4296a31fbb078d52e0750f64e881004a"


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


def resume_recipe_matches(saved: dict, current: dict) -> bool:
    """Accept the verified pre-sampling trainer without relaxing other guards."""
    previous = dict(saved)
    implementation = dict(previous.get("implementation", {}))
    if implementation.get("train_gpt2.py") == PRE_SAMPLE_TRAINER_SHA256:
        implementation["train_gpt2.py"] = current["implementation"]["train_gpt2.py"]
    previous["implementation"] = implementation
    return previous == current


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


def train(args, config):
    import numpy as np
    import torch
    from transformers import PreTrainedTokenizerFast

    config.validate()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CorpusError("No CUDA GPU. In Colab choose Runtime > Change runtime type > A100 GPU. Use --device cpu only for a tiny smoke test.")
        if not torch.cuda.is_bf16_supported():
            raise CorpusError("This training setup requires a BF16-capable GPU such as A100.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = "CPU (smoke tests only)"
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {hardware}", flush=True)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    run = resolve_run(args.corpus)
    corpus = inspect_corpus(run, args.allow_partial_corpus)
    settings = asdict(config)
    sampling_settings = {key: value for key, value in settings.items() if key.startswith("sample_")}
    training_settings = {key: value for key, value in settings.items() if not key.startswith("sample_")}
    recipe = {"config": training_settings, "data_version": DATA_VERSION, "corpus": corpus["fingerprint"],
              "precision": "bf16" if device.type == "cuda" else "fp32",
              "implementation": {name: file_sha256(Path(__file__).with_name(name)) for name in ("train_gpt2.py", "training_data.py")},
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "numpy")}}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_file = output / "run.json"
    if run_file.exists():
        if not args.resume:
            raise CorpusError("Output already contains a run. Use --resume or a new --output directory.")
        saved_run = json.loads(run_file.read_text())
        if not resume_recipe_matches(saved_run["recipe"], recipe):
            raise CorpusError("Corpus, settings, precision or library versions changed. Use a new output directory; resume cannot change the training objective or schedule.")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(output / "tokenizer", local_files_only=True)
        if {p.name: file_sha256(p) for p in (output / "tokenizer").iterdir() if p.is_file()} != saved_run["tokenizer_files"]:
            raise CorpusError("Saved tokenizer changed; refusing to resume.")
    else:
        if any(output.iterdir()):
            raise CorpusError("Output is not empty and has no run.json. Choose a fresh directory.")
        if args.resume:
            raise CorpusError("No run to resume. Omit --resume for the first session.")
        tokenizer = train_tokenizer(run, output / "tokenizer", config.vocab_size, config.min_frequency)
        tokenizer_files = {p.name: file_sha256(p) for p in (output / "tokenizer").iterdir() if p.is_file()}
        tokenizer_hash = sha256(json.dumps(tokenizer_files, sort_keys=True).encode())
        saved_run = {"recipe": recipe, "tokenizer_sha256": tokenizer_hash, "tokenizer_files": tokenizer_files, "corpus": corpus,
                     "identity": sha256(json.dumps({"recipe": recipe, "tokenizer": tokenizer_hash}, sort_keys=True).encode())}
        provenance = output / "provenance"
        provenance.mkdir()
        for name in ("train_gpt2.py", "training_data.py", "corpus_registry.py", "prepare_corpus.py"):
            shutil.copy2(Path(__file__).with_name(name), provenance / name)
        for name in ("report.json", "manifest.jsonl"):
            shutil.copy2(run / name, provenance / name)
        cleaning_report = json.loads((run / "report.json").read_text())
        for name, fallback, checksum in (("corpus-list.py", args.sources, "sources_sha256"), ("cleaning_rules.json", args.overrides, "rules_sha256")):
            source = run / name if (run / name).is_file() else fallback
            if source.is_file() and file_sha256(source) == cleaning_report[checksum]:
                shutil.copy2(source, provenance / name)
        write_json(run_file, saved_run)
    write_json(output / "sampling_config.json", sampling_settings)
    identity = saved_run["identity"]
    token_dir = args.work_dir / identity
    marker = token_dir / "complete.json"
    if not marker.exists():
        # If Colab lost /content, rebuild tokens with the frozen tokenizer on Drive.
        stats = tokenize_corpus(run, token_dir, tokenizer)
        write_json(marker, {"identity": identity, "index_sha256": file_sha256(token_dir / "index.json")})
    else:
        completion = json.loads(marker.read_text())
        if completion["identity"] != identity or completion["index_sha256"] != file_sha256(token_dir / "index.json"):
            raise CorpusError("Token cache metadata changed. Choose a fresh --work-dir to rebuild.")
        stats = json.loads((token_dir / "index.json").read_text())
        for split in SPLITS:
            if file_sha256(token_dir / f"{split}.bin") != stats[split]["sha256"]:
                raise CorpusError(f"Corrupt {split} token cache. Choose a fresh --work-dir to rebuild.")
    datasets = {split: TokenWindows(token_dir, split, config.context_length) for split in SPLITS}
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
    if args.resume and checkpoints:
        state = restore_checkpoint(checkpoints[-1], model, optimizer, scheduler, identity, device)
        print(f"Resuming optimizer step {state['step']}.", flush=True)
    elif args.resume:
        print("No complete checkpoint yet; restarting this initialized run from its seed.", flush=True)
        # Tokenizer training may use its own RNG; model initialization uses the seed above.
    summary = {"parameters": sum(p.numel() for p in model.parameters()), "vocab_size": len(tokenizer),
               "splits": {s: {"books": len(stats[s]["books"]), "target_tokens": stats[s]["target_tokens"], "windows": len(datasets[s])} for s in SPLITS},
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
            timed_out = args.max_seconds > 0 and elapsed >= args.max_seconds
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
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"), help="Preparation root or exact run directory")
    parser.add_argument("--output", type=Path, default=Path("runs/gpt2"), help="Persistent run/checkpoint directory; use Google Drive in Colab")
    parser.add_argument("--work-dir", type=Path, default=Path("data/tokenized"), help="Local token cache; may be rebuilt on resume")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("training_config.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume latest complete checkpoint in --output")
    parser.add_argument("--max-seconds", type=float, default=7200, help="Session training budget; checked after updates. Evaluation/checkpoint/export can run longer. 0 disables.")
    parser.add_argument("--max-steps", type=int, help="Override planned total optimizer steps (not extra steps on resume)")
    parser.add_argument("--allow-partial-corpus", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="First download/clean the current corpus list with raw caching")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--overrides", type=Path, default=Path(__file__).with_name("cleaning_rules.json"))
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("--max-seconds must be finite and nonnegative.")
    if args.prepare and args.resume:
        parser.error("Resume reuses its existing prepared corpus; do not combine --prepare and --resume.")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        config = TrainConfig(**json.loads(args.config.read_text()))
        if args.max_steps is not None:
            config.max_steps = args.max_steps
        config.validate()
        if args.prepare:
            previous = (args.corpus / "latest.json").read_bytes() if (args.corpus / "latest.json").exists() else None
            status = prepare_corpus.main(["--sources", str(args.sources), "--overrides", str(args.overrides), "--output", str(args.corpus), "--cache-raw"])
            latest = args.corpus / "latest.json"
            if not latest.exists() or latest.read_bytes() == previous:
                raise CorpusError("Preparation did not produce a new run; refusing to train on stale outputs.")
            if status and not args.allow_partial_corpus:
                raise CorpusError("Preparation needs review. Inspect its audit before training, or explicitly use --allow-partial-corpus.")
        train(args, config)
    except (CorpusError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")
# END UNCHANGED FILE: train_gpt2.py
print("Ready: train_gpt2.py")

# ============================================================================
# CELL 10: TEXT GENERATION — EXISTING CODE
# ============================================================================

# Run this cell to install the unchanged inference script.
# BEGIN UNCHANGED FILE: generate.py
(PROJECT / "generate.py").write_text(r'''#!/usr/bin/env python3
"""Generate text from a locally trained best_model export; no Hub downloads."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="The evening was quiet, and")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Token limit and temperature must be positive; top-p must be in (0, 1].")
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    model.config.use_cache = True
    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] == 0:
        inputs = {"input_ids": torch.tensor([[tokenizer.bos_token_id]]), "attention_mask": torch.ones((1, 1), dtype=torch.long)}
    if inputs["input_ids"].shape[1] + args.max_new_tokens > model.config.n_positions:
        parser.error(f"Prompt plus generated tokens must fit {model.config.n_positions} tokens. Shorten the prompt or --max-new-tokens.")
    with torch.inference_mode():
        result = model.generate(**{k: v.to(device) for k, v in inputs.items()}, max_new_tokens=args.max_new_tokens, use_cache=True, do_sample=True, temperature=args.temperature, top_p=args.top_p, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bad_words_ids=[[tokenizer.pad_token_id]])
    print(tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
''', encoding="utf-8")
# END UNCHANGED FILE: generate.py
print("Ready: generate.py")

# ============================================================================
# CELL 11: PREPARE THE CORPUS AND REVIEW THE AUDIT
# ============================================================================

# Downloads and cleaning start here. If the audit needs review, resolve the
# findings in Cell 3 and re-run with REBUILD_CORPUS=True, or deliberately set
# ALLOW_PARTIAL_CORPUS=True in Cell 1 to use accepted books only.
import hashlib

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def find_prepared_run():
    pointer = json.loads((CORPUS / 'latest.json').read_text())
    return CORPUS / pointer.get('relative_run', 'runs/' + Path(pointer['run']).name)

latest = CORPUS / 'latest.json'
if latest.exists() and not REBUILD_CORPUS:
    prepared = find_prepared_run()
    previous = json.loads((prepared / 'report.json').read_text())
    if (previous['sources_sha256'] != digest(PROJECT / 'corpus-list.py') or
        previous['rules_sha256'] != digest(PROJECT / 'cleaning_rules.json')):
        raise RuntimeError('List/rules changed. Set REBUILD_CORPUS=True and choose a new RUN_NAME.')
else:
    command = [sys.executable, str(PROJECT / 'prepare_corpus.py'),
               '--output', str(CORPUS), '--cache-raw']
    previous_pointer = latest.read_bytes() if latest.exists() else None
    completed = subprocess.run(command)
    if completed.returncode not in (0, 1) or not latest.exists():
        raise RuntimeError('Preparation failed before producing a run; inspect its output.')
    if latest.read_bytes() == previous_pointer:
        raise RuntimeError('Preparation produced no new run; refusing to use stale outputs.')
    prepared = find_prepared_run()
report = json.loads((prepared / 'report.json').read_text())
print(json.dumps(report['counts'], indent=2))
print('Audit:', prepared / 'audit.html')
manifest = [json.loads(line) for line in (prepared / 'manifest.jsonl').read_text().splitlines()]
for row in manifest:
    if row['status'] in ('needs_review', 'failed'):
        print(row['id'], row['status'], row.get('problems', row.get('error')))
if not report['complete'] and not ALLOW_PARTIAL_CORPUS:
    raise RuntimeError('Review audit.html and cleaning_rules.json, or explicitly set ALLOW_PARTIAL_CORPUS=True.')

# ============================================================================
# CELL 12: TRAIN OR RESUME — PROGRESS SAMPLES EVERY 100 OPTIMIZER UPDATES
# ============================================================================

# The original training command detects an existing checkpoint automatically.
# Watch this cell's output for the fixed-prompt continuations at steps 100, 200,
# 300, ... . They are also saved in OUTPUT / 'samples.jsonl'.
command = [sys.executable, str(PROJECT / 'train_gpt2.py'),
           '--corpus', str(prepared), '--output', str(OUTPUT),
           '--work-dir', '/content/gpt2-token-cache',
           '--config', str(PROJECT / 'training_config.json'),
           '--max-seconds', str(MAX_SESSION_SECONDS)]
if (OUTPUT / 'run.json').exists():
    command.append('--resume')
if ALLOW_PARTIAL_CORPUS:
    command.append('--allow-partial-corpus')
subprocess.run(command, check=True)
result = json.loads((OUTPUT / 'result.json').read_text())
print(json.dumps(result, indent=2))

# ============================================================================
# CELL 13: GENERATE FROM THE SAVED MODEL
# ============================================================================

# Run after a successful training/session stop. Edit PROMPT below to sample.
PROMPT = 'The evening was quiet, and'
subprocess.run([sys.executable, str(PROJECT / 'generate.py'),
                '--model', str(OUTPUT / 'best_model'), '--prompt', PROMPT,
                '--max-new-tokens', '200', '--temperature', '0.8', '--top-p', '0.95'], check=True)
