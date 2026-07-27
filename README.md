# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       34 |        2 |        0 |        0 |     94.12% |   329-331 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      140 |       13 |       32 |        6 |     88.95% |47, 51, 55, 84, 118, 283, 297, 305, 350, 366-369 |
| src/async\_batch\_llm/\_internal/capacity.py                 |      118 |       11 |       28 |        2 |     89.73% |42-43, 83-92, 158-159, 186-192 |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       18 |       28 |        9 |     72.12% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 159-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        6 |       16 |        1 |     90.91% |70, 76-77, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       56 |        4 |       10 |        3 |     89.39% |52-\>62, 54-\>62, 58-59, 117-119 |
| src/async\_batch\_llm/\_internal/guardrails.py               |      128 |        4 |       40 |        5 |     94.64% |59, 69, 98, 136-\>exit, 154 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      439 |       10 |      122 |       28 |     93.23% |175-\>exit, 177-\>exit, 186-\>exit, 194-\>exit, 291, 309-\>314, 328-\>exit, 399, 418, 425-\>435, 462, 558-\>561, 576-\>exit, 586, 717, 740-\>exit, 833, 869-\>874, 888-\>896, 936-\>938, 943-\>945, 966-\>971, 1051-\>1079, 1060, 1067-\>1069, 1080-\>1082, 1087, 1098-\>1100, 1178, 1185-\>1188 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      123 |        5 |       24 |        1 |     95.92% |100, 201, 262-268 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/artifacts.py                           |      461 |       72 |      150 |       31 |     82.49% |116, 127-\>129, 137, 139-141, 175-176, 237-238, 246-247, 254, 258-259, 278, 285, 291, 295, 300, 311, 313-316, 318, 341, 346, 361, 366-371, 377, 383, 400-401, 405, 461-462, 465-466, 468, 504, 520-523, 566-567, 592, 645-650, 659, 670, 702, 705, 741-742, 751, 765, 776, 782-\>794, 787-788, 816, 823-826, 847-848, 854 |
| src/async\_batch\_llm/base.py                                |      723 |       82 |      194 |       20 |     86.04% |107, 286, 290, 367-\>exit, 651-\>exit, 654-\>exit, 705, 742-\>744, 1019, 1021, 1188-1189, 1218-1219, 1230-1231, 1247-1257, 1270-1271, 1371, 1409, 1426, 1430-1435, 1439-\>exit, 1445, 1460-1463, 1467-\>exit, 1530, 1534, 1573-1581, 1593, 1595-1612, 1630-1646, 1714, 1760, 1779, 1783-\>1789, 1785-1788, 1804-\>exit, 1813-1817, 1834-1836 |
| src/async\_batch\_llm/callable\_strategy.py                  |      120 |        3 |       44 |        6 |     94.51% |49-\>exit, 62-\>exit, 81, 121, 160, 231-\>237 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       72 |       10 |       42 |        6 |     85.96% |68, 70, 72, 98-99, 107, 117, 183-184, 272 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      180 |        2 |       90 |        3 |     98.15% |200, 420, 436-\>exit |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       45 |        0 |       22 |        2 |     97.01% |63-\>exit, 74-\>exit |
| src/async\_batch\_llm/gateway.py                             |       72 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      146 |        9 |       34 |        6 |     91.67% |31-32, 66-\>64, 68, 83-85, 299-\>exit, 309-\>exit, 377, 635-637 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        1 |        0 |        0 |     90.91% |        98 |
| src/async\_batch\_llm/models.py                              |      543 |       33 |      222 |       28 |     91.76% |40-42, 45-46, 256-\>259, 336-\>339, 344, 347, 367, 512-\>515, 517, 523-\>exit, 548, 627-\>630, 631, 634, 647-651, 687-688, 704-\>734, 715, 746, 780, 1052-\>1101, 1055-\>1058, 1059-\>1095, 1075-1076, 1081, 1102-\>1106, 1104-1105, 1112, 1156-\>1166, 1159-\>1166, 1161-1162, 1180, 1183-\>exit, 1387-1388, 1577-\>1582 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       23 |        1 |        0 |        0 |     95.65% |        61 |
| src/async\_batch\_llm/observers/metrics.py                   |      107 |        8 |       46 |        9 |     87.58% |42, 58-\>65, 68-\>74, 75-\>exit, 91, 92-\>exit, 99-104, 107, 113-\>exit |
| src/async\_batch\_llm/parallel.py                            |      373 |       30 |      108 |        9 |     91.89% |79-80, 83-84, 150, 190, 306, 310, 321, 325, 329, 333, 337, 376, 484-485, 500, 513-516, 526-528, 600-\>603, 646, 651, 656, 758, 815-\>820, 844-\>849, 896, 900-901 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      239 |       39 |       96 |       21 |     81.49% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 313-314, 329, 339-340, 351, 422, 429, 436, 441, 459, 476, 512, 543-546, 598-599, 610-611, 617, 620-621, 633, 644 |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      687 |       90 |      202 |       41 |     83.91% |84-85, 121-\>exit, 129, 134, 209-210, 212, 220-221, 260-263, 288, 298, 330-331, 363-368, 373, 377, 407, 416-417, 474-477, 497, 529, 548-551, 554-555, 557-\>564, 560-561, 572, 601, 609, 623, 629, 635-636, 658, 660-\>exit, 674-\>676, 677, 683-\>685, 694, 702-708, 745-746, 757-\>756, 772-\>771, 786-787, 794-\>804, 799, 809-811, 827-\>842, 856, 876, 893-896, 900-902, 995-997, 1013, 1080, 1087, 1090, 1094, 1105, 1136, 1224-1225, 1290-1291, 1337-1338, 1346-1347, 1350-1351, 1355 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      113 |       12 |       26 |        1 |     90.65% |62-73, 130-\>exit, 375-376 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      190 |        1 |       72 |        4 |     98.09% |63-\>exit, 195-\>197, 251-\>254, 313, 319-\>exit |
| src/async\_batch\_llm/token\_extractor.py                    |       63 |        3 |       30 |        5 |     91.40% |51-\>61, 68-\>72, 76-\>89, 114, 132-\>135, 151-152 |
| **TOTAL**                                                    | **5711** |  **484** | **1800** |  **255** | **89.47%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/geoff-davis/async-batch-llm/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeoff-davis%2Fasync-batch-llm%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.