import random

from src.data.core.smart_dataset import SmartDataset
from src.data.pinterest_faces import PinterestFacesDS, PinterestFaces


class BackdoorData(object):
    attack_dataset: SmartDataset
    test_dataset: SmartDataset


class PinterestBackdoorData(BackdoorData):
    # Ordered decreasingly by number of images, randing from 236 to 86
    CANDIDATES = ['Leonardo Dicaprio', 'Katherine Langford', 'Alexandra Daddario', 'Elizabeth Olsen', 'Margot Robbie',
                  'Amber Heard', 'Logan Lerman', 'Alycia Dabnem Carey', 'Brenton Thwaites', 'Emilia Clarke',
                  'Megan Fox', 'Sophie Turner', 'Kiernen Shipka', 'Anne Hathaway', 'Scarlett Johansson', 'Gal Gadot',
                  'Tom Hardy', 'Andy Samberg', 'Natalie Dormer', 'Barbara Palvin', 'Henry Cavil', 'Maisie Williams',
                  'Madelaine Petsch', 'Millie Bobby Brown', 'Zac Efron', 'Tom Holland', 'Ellen Page', 'Selena Gomez',
                  'Jason Momoa', 'Zoe Saldana', 'Grant Gustin', 'Danielle Panabaker', 'Jennifer Lawrence', 'Tom Hiddleston',
                  'Wentworth Miller', 'Hugh Jackman', 'Miley Cyrus', 'Rebecca Ferguson', 'Tom Ellis', 'Katharine Mcphee',
                  'Mark Ruffalo', 'Chris Pratt', 'Morena Baccarin', 'Krysten Ritter', 'Penn Badgley', 'Brie Larson',
                  'Lindsey Morgan', 'Ursula Corbero', 'Chris Evans', 'Jeremy Renner', 'Natalie Portman', 'Camila Mendes',
                  'Eliza Taylor', 'Marie Avgeropoulos', 'Rami Malek', 'Chris Hemsworth', 'Jake Mcdorman', 'Sarah Wayne Callies',
                  'Stephen Amell', 'Irina Shayk', 'Elizabeth Lail', 'Shakira Isabel Mebarak', 'Melissa Fumero', 'Alex Lawther',
                  'Lili Reinhart', 'Richard Harmon', 'Dominic Purcell', 'Jessica Barden', 'Alvaro Morte', 'Bobby Morley',
                  'Emma Stone', 'Zendaya', 'Elon Musk', 'Nadia Hilker', 'Tuppence Middleton', 'Rihanna', 'Taylor Swift',
                  'Inbar Lavi', 'Pedro Alonso', 'Anthony Mackie', 'Maria Pedraza', 'Barack Obama', 'Amanda Crew',
                  'Josh Radnor', 'Neil Patrick Harris', 'Jimmy Fallon', 'Brian J. Smith', 'Jeff Bezos', 'Cristiano Ronaldo',
                   'Mark Zuckerberg', 'Lionel Messi']

    def __init__(self, name=None, dataset: PinterestFacesDS = None):
        super(PinterestBackdoorData, self).__init__()
        self.name = name or random.choice(PinterestBackdoorData.CANDIDATES)
        dataset = dataset or PinterestFaces.load('fit').fit_dataset

        class_ = dataset.class_to_idx[self.name]
        self.attack_dataset, self.test_dataset = dataset.by_class[class_].fraction_random_split(0.1)
