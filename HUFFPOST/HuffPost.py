# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuffPost Dataset."""


import csv
import json
import os

import datasets



_CITATION = """\
@book{book,
  author = {Misra, Rishabh and Grover, Jigyasa},
  year = {2021},
  month = {01},
  pages = {},
  title = {Sculpting Data for ML: The first act of Machine Learning},
  isbn = {978-0-578-83125-1}
}

@dataset{dataset,
  author = {Misra, Rishabh},
  year = {2018},
  month = {06},
  pages = {},
  title = {News Category Dataset},
  doi = {10.13140/RG.2.2.20331.18729}
}
"""


_DESCRIPTION = """\
A dataset of approximately 200K news headlines from the year 2012 to 2018 collected from HuffPost."""

_HOMEPAGE = "https://www.kaggle.com/datasets/rmisra/news-category-dataset"

_LICENSE = "CC0: Public Domain"

_URLS = "https://huggingface.co/datasets/khalidalt/HuffPost/resolve/main/News_Category_Dataset_v2.json"



class HuffPost(datasets.GeneratorBasedBuilder):
    """HuffPost Dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description="Default config"),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        
        features = datasets.Features(
            {
                "category": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "authors": datasets.Value("string"),
                "link": datasets.Value("string"),
                "short_description": datasets.Value("string"),
                "date": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["POLITICS","WELLNESS","ENTERTAINMENT","TRAVEL","STYLE & BEAUTY",
                                                    "PARENTING","HEALTHY LIVING","QUEER VOICES","FOOD & DRINK",
                                                    "BUSINESS","COMEDY","SPORTS","BLACK VOICES","HOME & LIVING","PARENTS",
                                                    "THE WORLDPOST","WEDDINGS","WOMEN","IMPACT","DIVORCE","CRIME","MEDIA",
                                                    "WEIRD NEWS","GREEN","WORLDPOST","RELIGION","STYLE","SCIENCE",
                                                    "WORLD NEWS","TASTE","TECH","MONEY","ARTS","FIFTY","GOOD NEWS",
                                                    "ARTS & CULTURE","ENVIRONMENT","COLLEGE","LATINO VOICES","CULTURE & ARTS",
                                                    "EDUCATION"]),
            }
            )

        return datasets.DatasetInfo(

            description=_DESCRIPTION,
            features=features,  
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir},
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                data['label'] = data['category'] 
                # Yields examples as (key, example) tuples
                yield key, data
