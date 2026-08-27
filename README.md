# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/async-batch-llm/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/async\_batch\_llm/\_\_init\_\_.py                        |       35 |        2 |        0 |        0 |     94.29% |   343-345 |
| src/async\_batch\_llm/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/admission.py                |      391 |       20 |      114 |       17 |     92.67% |93, 97, 105, 122, 124, 126, 146, 162, 164, 250, 282-283, 367, 381, 386-387, 444-\>447, 447-\>exit, 468-\>466, 532, 573, 583-584 |
| src/async\_batch\_llm/\_internal/artifact\_codec.py          |      140 |       13 |       32 |        6 |     88.95% |47, 51, 55, 84, 118, 283, 297, 305, 350, 366-369 |
| src/async\_batch\_llm/\_internal/capacity.py                 |      118 |       11 |       28 |        2 |     89.73% |42-43, 83-92, 158-159, 186-192 |
| src/async\_batch\_llm/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/\_internal/error\_logging.py           |       76 |       16 |       28 |        8 |     75.00% |36-\>39, 87, 101-\>111, 109-110, 112, 115-126, 143-\>164, 145, 147-150, 161-162 |
| src/async\_batch\_llm/\_internal/event\_dispatcher.py        |       61 |        6 |       16 |        1 |     90.91% |70, 76-77, 110, 124, 126 |
| src/async\_batch\_llm/\_internal/executor\_host.py           |       63 |       11 |        6 |        3 |     79.71% |94-98, 156-157, 160-161, 163-166, 169 |
| src/async\_batch\_llm/\_internal/guardrails.py               |      136 |        9 |       44 |        5 |     91.11% |59, 69, 98, 137-\>exit, 155, 167-171 |
| src/async\_batch\_llm/\_internal/item\_executor.py           |      614 |       25 |      196 |       39 |     92.10% |198-\>200, 232-\>exit, 234-\>exit, 243-\>exit, 251-\>exit, 275, 291, 360, 420, 455-458, 465-466, 478, 537, 564-\>569, 608-\>613, 627-\>exit, 699, 718, 725-\>735, 763, 839-\>842, 857-\>exit, 867, 998, 1021-\>exit, 1131, 1147-\>1152, 1192-1193, 1196-\>1202, 1234-\>1242, 1282-\>1286, 1291-\>1295, 1330-\>1359, 1349-\>1359, 1370-\>1381, 1381-\>1387, 1389-\>1400, 1496, 1503-\>1505, 1517, 1519, 1521-\>1523, 1528, 1539-\>1541, 1636, 1643-\>1646 |
| src/async\_batch\_llm/\_internal/rate\_limit\_coordinator.py |      135 |        5 |       32 |        1 |     96.41% |102, 208, 280-286 |
| src/async\_batch\_llm/\_internal/strategy\_lifecycle.py      |       40 |        2 |       10 |        1 |     94.00% |79-\>78, 100, 104 |
| src/async\_batch\_llm/artifacts.py                           |      461 |       72 |      150 |       32 |     82.32% |116, 127-\>129, 137, 139-141, 175-176, 237-238, 246-247, 254, 258-259, 278, 285, 291, 295, 300, 311, 313-316, 318, 341, 346, 361, 366-371, 377, 383, 400-401, 405, 461-462, 465-466, 468, 504, 520-523, 566-567, 592, 645-650, 659, 670, 676-\>675, 702, 705, 741-742, 751, 765, 776, 782-\>794, 787-788, 816, 823-826, 847-848, 854 |
| src/async\_batch\_llm/base.py                                |      787 |       82 |      212 |       21 |     87.09% |107, 293, 302, 379-\>exit, 668-\>exit, 671-\>exit, 722, 759-\>761, 814-\>819, 1096, 1098, 1311-1312, 1341-1342, 1353-1354, 1370-1380, 1393-1394, 1494, 1532, 1549, 1553-1558, 1562-\>exit, 1568, 1583-1586, 1590-\>exit, 1653, 1657, 1696-1704, 1716, 1718-1735, 1753-1769, 1837, 1883, 1902, 1906-\>1912, 1908-1911, 1927-\>exit, 1936-1940, 1957-1959 |
| src/async\_batch\_llm/callable\_strategy.py                  |      132 |        4 |       48 |        7 |     93.89% |50-\>exit, 63-\>exit, 82, 122, 163, 222, 262-\>268 |
| src/async\_batch\_llm/classifiers/\_\_init\_\_.py            |        4 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/classifiers/gemini.py                  |       69 |        9 |       38 |        5 |     86.92% |51, 53, 55, 84-85, 119-120, 222, 230 |
| src/async\_batch\_llm/classifiers/openai.py                  |       76 |       10 |       46 |        6 |     86.89% |70, 72, 74, 116-117, 125, 135, 201-202, 290 |
| src/async\_batch\_llm/classifiers/openrouter.py              |       19 |        0 |        8 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/core/config.py                         |      184 |        3 |       92 |        3 |     97.83% |205, 440, 448 |
| src/async\_batch\_llm/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/factory.py                             |       45 |        0 |       22 |        2 |     97.01% |63-\>exit, 74-\>exit |
| src/async\_batch\_llm/gateway.py                             |       72 |        0 |       14 |        0 |    100.00% |           |
| src/async\_batch\_llm/llm\_strategies.py                     |      180 |       13 |       44 |        8 |     90.62% |32-33, 67-\>65, 69, 84-86, 274-275, 325-\>exit, 337-\>exit, 420, 438, 605-\>607, 610, 742-744 |
| src/async\_batch\_llm/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/models.py                              |      784 |      104 |      346 |       59 |     82.74% |48-50, 53-54, 283-\>286, 363-\>366, 371, 374, 394, 539-\>542, 544, 550-\>exit, 575, 654-\>657, 658, 661, 674-678, 714-715, 731-\>761, 742, 773, 807, 1079-\>1128, 1082-\>1085, 1086-\>1122, 1102-1103, 1108, 1129-\>1133, 1131-1132, 1176-\>1186, 1179-\>1186, 1181-1182, 1200, 1203-\>exit, 1402-1403, 1489, 1512, 1522-1523, 1526, 1538-\>1547, 1542-1546, 1552-1560, 1567, 1641, 1650, 1685, 1688, 1690-1694, 1771-\>1773, 1779-1780, 1817, 1819, 1851-\>1853, 1869, 1871, 1881, 1887-1900, 1902-1904, 1911-1919, 1921, 1936-1937, 1947, 1958-\>1960, 1961-\>1963, 1964-\>1966, 1966-\>1973, 1976-1987, 1995, 1997, 2010-\>2015 |
| src/async\_batch\_llm/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/async\_batch\_llm/observers/metrics.py                   |      139 |       12 |       70 |       14 |     85.65% |55, 72-\>79, 82-\>88, 89-\>exit, 114-\>108, 117-\>exit, 128, 134-135, 141, 142-\>147, 152, 155-160, 163, 169-\>exit |
| src/async\_batch\_llm/parallel.py                            |      375 |       38 |      108 |       11 |     89.86% |80-81, 84-85, 152, 192, 295, 299, 310, 314, 322, 326, 330, 334, 338, 373-374, 378-379, 381, 386, 452-453, 456, 498-499, 531-534, 564-\>567, 627, 632, 637, 739, 796-\>801, 825-\>830, 877, 938-939, 941 |
| src/async\_batch\_llm/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/async\_batch\_llm/provider\_output.py                    |      102 |        1 |       28 |        1 |     98.46% |       225 |
| src/async\_batch\_llm/serialization.py                       |      249 |       41 |      102 |       23 |     81.20% |113, 116, 130, 139-140, 170, 222-223, 227, 252, 258, 268, 270, 350-351, 366, 376-377, 386, 388, 402, 473, 480, 487, 492, 510, 527, 563, 594-597, 649-650, 661-662, 668, 671-672, 684, 695 |
| src/async\_batch\_llm/single.py                              |       27 |        3 |        4 |        1 |     87.10% | 52-53, 67 |
| src/async\_batch\_llm/sqlite\_artifacts.py                   |      687 |       89 |      202 |       40 |     84.14% |84-85, 121-\>exit, 129, 209-210, 212, 220-221, 260-263, 288, 298, 330-331, 363-368, 373, 377, 407, 416-417, 474-477, 497, 529, 548-551, 554-555, 557-\>564, 560-561, 572, 601, 609, 623, 629, 635-636, 658, 660-\>exit, 674-\>676, 677, 683-\>685, 694, 702-708, 745-746, 757-\>756, 772-\>771, 786-787, 794-\>804, 799, 809-811, 827-\>842, 856, 876, 893-896, 900-902, 995-997, 1013, 1080, 1087, 1090, 1094, 1105, 1136, 1224-1225, 1290-1291, 1337-1338, 1346-1347, 1350-1351, 1355 |
| src/async\_batch\_llm/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/async\_batch\_llm/strategies/errors.py                   |      131 |       16 |       34 |        5 |     87.27% |62-73, 148-\>exit, 181, 200, 384, 392, 446-447 |
| src/async\_batch\_llm/strategies/rate\_limit.py              |       31 |        0 |        2 |        0 |    100.00% |           |
| src/async\_batch\_llm/streaming.py                           |      190 |        0 |       72 |        4 |     98.47% |63-\>exit, 195-\>197, 251-\>254, 319-\>exit |
| src/async\_batch\_llm/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/async\_batch\_llm/token\_extractor.py                    |      111 |       10 |       48 |        9 |     86.79% |73-\>81, 88-\>92, 96-\>109, 119, 149, 162, 173-\>176, 185-188, 214, 235, 246-247 |
| **TOTAL**                                                    | **6849** |  **628** | **2222** |  **334** | **88.42%** |           |


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