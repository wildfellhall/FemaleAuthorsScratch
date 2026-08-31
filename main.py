"""
Not completed yet, but the script should be pretty similar code-wise to my Poetry50M repository. 
As always, a line of hashtags denotes the presence of a separate cell. 
"""
# Import/installation cell: 
!pip install -q transformers datasets tokenizers accelerate torch

import os
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

#####################################################################

# Sources: 
sources = [
    "https://www.gutenberg.org/files/1342/1342-0.txt", #Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/files/3268/3268-0.txt", #The Mysteries of Udolpho by Ann Ward Radcliffe
    "https://www.gutenberg.org/files/21839/21839-0.txt", #Sense and Sensibility by Jane Austen
    "https://www.gutenberg.org/files/121/121-0.txt", #Northanger Abbey by Jane Austen
    "https://www.gutenberg.org/files/158/158-0.txt", #Emma by Jane Austen
    "https://www.gutenberg.org/files/105/105-0.txt", #Persuasion by Jane Austen
    "https://www.gutenberg.org/files/141/141-0.txt", #Mansfield Park by Jane Austen
    "https://www.gutenberg.org/files/145/145-0.txt", #Middlemarch by George Eliot
    "https://www.gutenberg.org/files/550/550-0.txt", #Silas Marner by George Eliot
    "https://www.gutenberg.org/files/507/507-0.txt", #Adam Bede by George Eliot
    "https://www.gutenberg.org/files/6688/6688-0.txt", #The Mill on the Floss by George Eliot
    "https://www.gutenberg.org/files/7469/7469-0.txt", #Daniel Deronda by George Eliot
    "https://www.gutenberg.org/files/24020/24020-0.txt", #Romola by George Eliot
    "https://www.gutenberg.org/files/969/969-0.txt", #The Tenant of Wildfell Hall by Anne Brontë
    "https://www.gutenberg.org/files/767/767-0.txt", #Agnes Grey by Anne Brontë
    "https://www.gutenberg.org/files/1028/1028-0.txt", #The Professor by Charlotte Brontë
    "https://www.gutenberg.org/files/30486/30486-0.txt", #Shirley by Charlotte Brontë
    "https://www.gutenberg.org/files/9182/9182-0.txt", #Villette by Charlotte Brontë
    "https://www.gutenberg.org/files/1260/1260-0.txt", #Jane Eyre: An Autobiography by Charlotte Brontë
    "https://www.gutenberg.org/files/768/768-0.txt", #Wuthering Heights by Emily Brontë
    "https://www.gutenberg.org/files/37106/37106-0.txt", #Little Women; Or, Meg, Jo, Beth, and Amy by Louisa May Alcott
    "https://www.gutenberg.org/files/28203/28203-0.txt", #Moods by Louisa May Alcott
    "https://www.gutenberg.org/files/41127/41127-0.txt", #Rose in Bloom by Louisa May Alcott
    "https://www.gutenberg.org/files/2787/2787-0.txt", #An Old-Fashioned Girl by Louisa May Alcott
    "https://www.gutenberg.org/files/2726/2726-0.txt", #Eight Cousins by Louisa May Alcott
    "https://www.gutenberg.org/files/2786/2786-0.txt", #Jack and Jill by Louisa May Alcott
    "https://www.gutenberg.org/files/2788/2788-0.txt", #Little Men: Life at Plumfield With Jo's Boys by Louisa May Alcott
    "https://www.gutenberg.org/files/3499/3499-0.txt", #Jo's Boys by Louisa May Alcott
    "https://www.gutenberg.org/files/84/84-0.txt", #Frankenstein; or, the modern prometheus by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/18247/18247-0.txt", #The Last Man by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/15238/15238-0.txt", #Mathilda by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/64329/64329-0.txt", #Falkner: A Novel by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/394/394-0.txt", #Cranford by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4276/4276-0.txt", #North and South by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2153/2153-0.txt", #Mary Barton by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4274/4274-0.txt", #Wives and Daughters by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4275/4275-0.txt", #Ruth by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2524/2524-0.txt", #My Lady Ludlow by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4537/4537-0.txt", #Sylvia's Lovers — Complete by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/4268/4268-0.txt", #Cousin Phillis by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/2522/2522-0.txt", #A Dark Night's Work by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/7371/7371-0.txt", #A Sicilian Romance by Ann Ward Radcliffe
    "https://www.gutenberg.org/files/64701/64701-0.txt", #The Romance of the Forest, interspersed with some pieces of poetry. by Radcliffe
    "https://www.gutenberg.org/files/66545/66545-0.txt", #The Heir of Mondolfo by Mary Wollstonecraft Shelley
    "https://www.gutenberg.org/files/31180/31180-0.txt", #Ellen Middleton—A Tale by Georgiana Fullerton
    "https://www.gutenberg.org/files/3322/3322-0.txt", #East Lynne by Mrs. Henry Wood
    "https://www.gutenberg.org/files/15627/15627-0.txt", #Verner's Pride by Mrs. Henry Wood
    "https://www.gutenberg.org/files/34587/34587-0.txt", #Mrs. Halliburton's Troubles by Mrs. Henry Wood
    "https://www.gutenberg.org/files/22121/22121-0.txt", #Olive: A Novel by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/45975/45975-0.txt", #The Little Lame Prince and His Travelling Cloak by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/21767/21767-0.txt", #Agatha's Husband: A Novel by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/2351/2351-0.txt", #John Halifax, Gentleman by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/13461/13461-0.txt", #Mistress and Maid: A Household Story by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/14708/14708-0.txt", #The Laurel Bush: An Old-Fashioned Love Story by Dinah Maria Mulock Craik
    "https://www.gutenberg.org/files/36157/36157-0.txt", #Daisy Burns (Volume 1) by Julia Kavanagh
    "https://www.gutenberg.org/files/36158/36158-0.txt", #Daisy Burns (Volume 2) by Julia Kavanagh
    "https://www.gutenberg.org/files/36160/36160-0.txt", #Rachel Gray: A Tale Founded on Fact by Julia Kavanagh
    "https://www.gutenberg.org/files/1952/1952-0.txt", #The Yellow Wallpaper by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/3016/3016-0.txt", #What Diantha Did by Charlotte Perkins Gilman
    "https://www.gutenberg.org/files/6053/6053-0.txt", #Evelina, Or, the History of a Young Lady's Entrance into the World by Fanny Burney
    "https://www.gutenberg.org/files/40619/40619-0.txt", #Camilla; or, A Picture of Youth by Fanny Burney
    "https://www.gutenberg.org/files/171/171-0.txt", #Charlotte Temple by Mrs. Rowson
    "https://www.gutenberg.org/files/5182/5182-0.txt", #The Old English Baron: a Gothic Story by Clara Reeve
    "https://www.gutenberg.org/files/23810/23810-0.txt", #At Fault by Kate Chopin
    "https://www.gutenberg.org/files/6346/6346-0.txt", #Cecilia; Or, Memoirs of an Heiress — Volume 1 by Fanny Burney
    "https://www.gutenberg.org/files/7146/7146-0.txt", #Cecilia; Or, Memoirs of an Heiress — Volume 2 by Fanny Burney
    "https://www.gutenberg.org/files/7152/7152-0.txt", #Cecilia; Or, Memoirs of an Heiress — Volume 3 by Fanny Burney
    "https://www.gutenberg.org/files/37437/37437-0.txt", #The Wanderer; or, Female Difficulties (Volume 1 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37438/37438-0.txt", #The Wanderer; or, Female Difficulties (Volume 2 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37439/37439-0.txt", #The Wanderer; or, Female Difficulties (Volume 3 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37440/37440-0.txt", #The Wanderer; or, Female Difficulties (Volume 4 of 5) by Fanny Burney
    "https://www.gutenberg.org/files/37441/37441-0.txt", #The Wanderer; or, Female Difficulties (Volume 5 of 5) by Fanny Burney
]
