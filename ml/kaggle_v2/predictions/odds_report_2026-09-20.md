# Model picks vs bookmaker odds, 2026-09-20
21 races, 187 runners (0 without odds). Finished races use the closing SP, the rest the best price across bookmakers right now. Break-even odds = 1 / backtest precision of runners at or above the pick's confidence band (>=0.10: 5.41, >=0.15: 4.13, >=0.20: 3.31, >=0.25: 2.78, >=0.30: 2.45, >=0.35: 2.14, >=0.40: 1.90, >=0.50: 1.60).

## #1 pick per race
| course | off | horse | p | fair_odds | odds | price_kind | BE_odds | bettable | edge | market_p | n_books | won |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hamilton | 14:00 | Celtic Storm (FR) | 0.356 | 2.813 | 2.020 | best now | 2.140 | False | -0.282 | 0.452 | 29 | nan |
| Hamilton | 14:30 | Prosperity Angel (GB) | 0.223 | 4.484 | 6.500 | best now | 3.314 | True | 0.450 | 0.149 | 29 | nan |
| Hamilton | 15:00 | Doon The Glen (GB) | 0.202 | 4.948 | 7.000 | best now | 3.314 | True | 0.415 | 0.141 | 29 | nan |
| Hamilton | 15:30 | Thunderstorm Katie (GB) | 0.160 | 6.234 | 4.800 | best now | 4.132 | True | -0.230 | 0.198 | 29 | nan |
| Hamilton | 16:00 | Time Loop (IRE) | 0.176 | 5.685 | 5.800 | best now | 4.132 | True | 0.020 | 0.158 | 29 | nan |
| Hamilton | 16:30 | Motawaared (GB) | 0.214 | 4.682 | 4.100 | best now | 3.314 | True | -0.124 | 0.225 | 29 | nan |
| Hamilton | 17:00 | Dandy's Angel (IRE) | 0.168 | 5.936 | 10.500 | best now | 4.132 | True | 0.769 | 0.078 | 29 | nan |
| Listowel | 14:18 | Sonny Corleone (IRE) | 0.213 | 4.696 | 6.200 | best now | 3.314 | True | 0.320 | 0.175 | 28 | nan |
| Listowel | 14:48 | Le Nez Creux (FR) | 0.351 | 2.848 | 5.400 | best now | 2.140 | True | 0.896 | 0.221 | 28 | nan |
| Listowel | 15:18 | Wonderful Everyday (IRE) | 0.119 | 8.411 | 6.200 | best now | 5.414 | True | -0.263 | 0.130 | 28 | nan |
| Listowel | 15:48 | Haveanothertry (IRE) | 0.264 | 3.789 | 13.500 | best now | 2.783 | True | 2.563 | 0.074 | 28 | nan |
| Listowel | 16:18 | Ocastle Des Mottes (FR) | 0.298 | 3.354 | 2.500 | best now | 2.783 | False | -0.255 | 0.371 | 28 | nan |
| Listowel | 16:48 | Jalila Moriviere (FR) | 0.117 | 8.577 | 4.600 | best now | 5.414 | False | -0.464 | 0.163 | 28 | nan |
| Listowel | 17:18 | Green Hint (IRE) | 0.305 | 3.283 | 1.840 | best now | 2.451 | False | -0.439 | 0.435 | 28 | nan |
| Plumpton | 14:08 | Getarose (IRE) | 0.407 | 2.457 | 3.150 | best now | 1.902 | True | 0.282 | 0.302 | 29 | nan |
| Plumpton | 14:38 | Premier Tenor (FR) | 0.439 | 2.278 | 1.740 | best now | 1.902 | False | -0.236 | 0.326 | 22 | nan |
| Plumpton | 15:08 | Premier Fantasy (IRE) | 0.409 | 2.443 | 2.120 | best now | 1.902 | True | -0.132 | 0.441 | 28 | nan |
| Plumpton | 15:38 | Mancero (FR) | 0.265 | 3.773 | 4.100 | best now | 2.783 | True | 0.087 | 0.211 | 29 | nan |
| Plumpton | 16:08 | Edelak (IRE) | 0.280 | 3.577 | 5.500 | best now | 2.783 | True | 0.537 | 0.172 | 22 | nan |
| Plumpton | 16:38 | Farhh Echo (IRE) | 0.233 | 4.301 | 2.800 | best now | 3.314 | False | -0.349 | 0.326 | 29 | nan |
| Plumpton | 17:08 | Shane's Spirit (IRE) | 0.522 | 1.915 | 1.860 | best now | 1.598 | True | -0.029 | 0.503 | 29 | nan |

**#1 picks at a bettable price: 15 of 21 (71%)**; with model edge p x odds - 1 >= 0: 10 of 21.
Races finished so far: 0 of 21.

## Every runner with p >= 0.15
| course | off | horse | blend_all | price | break_even | bettable | edge |
|---|---|---|---|---|---|---|---|
| Hamilton | 14:00 | Celtic Storm (FR) | 0.356 | 2.020 | 2.140 | False | -0.282 |
| Hamilton | 14:00 | Charmed Boy (IRE) | 0.184 | 6.800 | 4.132 | True | 0.252 |
| Hamilton | 14:30 | Prosperity Angel (GB) | 0.223 | 6.500 | 3.314 | True | 0.450 |
| Hamilton | 14:30 | Pj's Corner (GB) | 0.187 | 5.400 | 4.132 | True | 0.012 |
| Hamilton | 14:30 | Space Dreamer (IRE) | 0.176 | 4.900 | 4.132 | True | -0.138 |
| Hamilton | 15:00 | Doon The Glen (GB) | 0.202 | 7.000 | 3.314 | True | 0.415 |
| Hamilton | 15:00 | Fear And Fast (GB) | 0.187 | 6.400 | 4.132 | True | 0.194 |
| Hamilton | 15:00 | Kilmac Air (IRE) | 0.184 | 7.000 | 4.132 | True | 0.286 |
| Hamilton | 15:30 | Thunderstorm Katie (GB) | 0.160 | 4.800 | 4.132 | True | -0.230 |
| Hamilton | 15:30 | Persian Spirit (IRE) | 0.159 | 6.800 | 4.132 | True | 0.084 |
| Hamilton | 16:00 | Time Loop (IRE) | 0.176 | 5.800 | 4.132 | True | 0.020 |
| Hamilton | 16:30 | Motawaared (GB) | 0.214 | 4.100 | 3.314 | True | -0.124 |
| Hamilton | 17:00 | Dandy's Angel (IRE) | 0.168 | 10.500 | 4.132 | True | 0.769 |
| Hamilton | 17:00 | Uncle Liam (IRE) | 0.163 | 3.350 | 4.132 | False | -0.454 |
| Listowel | 14:18 | Sonny Corleone (IRE) | 0.213 | 6.200 | 3.314 | True | 0.320 |
| Listowel | 14:18 | Minella Buoy (GB) | 0.213 | 2.520 | 3.314 | False | -0.463 |
| Listowel | 14:18 | Danny Power (IRE) | 0.178 | 4.800 | 4.132 | True | -0.144 |
| Listowel | 14:48 | Le Nez Creux (FR) | 0.351 | 5.400 | 2.140 | True | 0.896 |
| Listowel | 14:48 | Tumbling In (IRE) | 0.309 | 2.780 | 2.451 | True | -0.140 |
| Listowel | 14:48 | Easter Bonnet (FR) | 0.181 | 6.200 | 4.132 | True | 0.125 |
| Listowel | 15:48 | Haveanothertry (IRE) | 0.264 | 13.500 | 2.783 | True | 2.563 |
| Listowel | 15:48 | Gillane (IRE) | 0.187 | 9.000 | 4.132 | True | 0.683 |
| Listowel | 15:48 | Run Ted Run (IRE) | 0.175 | 1.630 | 4.132 | False | -0.715 |
| Listowel | 16:18 | Ocastle Des Mottes (FR) | 0.298 | 2.500 | 2.783 | False | -0.255 |
| Listowel | 16:18 | Apple's Of Bresil (FR) | 0.208 | 4.600 | 3.314 | True | -0.045 |
| Listowel | 16:18 | Luker's Tipple (IRE) | 0.200 | 12.000 | 3.314 | True | 1.404 |
| Listowel | 16:18 | Sky Lord (GB) | 0.177 | 3.900 | 4.132 | False | -0.308 |
| Listowel | 17:18 | Green Hint (IRE) | 0.305 | 1.840 | 2.451 | False | -0.439 |
| Plumpton | 14:08 | Getarose (IRE) | 0.407 | 3.150 | 1.902 | True | 0.282 |
| Plumpton | 14:08 | Siorai (IRE) | 0.351 | 2.040 | 2.140 | False | -0.284 |
| Plumpton | 14:08 | Galeforcechopper (GB) | 0.242 | 5.000 | 3.314 | True | 0.209 |
| Plumpton | 14:38 | Premier Tenor (FR) | 0.439 | 1.740 | 1.902 | False | -0.236 |
| Plumpton | 14:38 | Arcturus Flame (IRE) | 0.314 | 1.490 | 2.451 | False | -0.533 |
| Plumpton | 14:38 | Raulin (IRE) | 0.168 | 3.400 | 4.132 | False | -0.429 |
| Plumpton | 15:08 | Premier Fantasy (IRE) | 0.409 | 2.120 | 1.902 | True | -0.132 |
| Plumpton | 15:08 | One Dimensional (GB) | 0.209 | 4.100 | 3.314 | True | -0.143 |
| Plumpton | 15:38 | Mancero (FR) | 0.265 | 4.100 | 2.783 | True | 0.087 |
| Plumpton | 15:38 | French Diablo (FR) | 0.186 | 3.200 | 4.132 | False | -0.404 |
| Plumpton | 15:38 | Jefe Triunfo (FR) | 0.181 | 4.800 | 4.132 | True | -0.130 |
| Plumpton | 15:38 | Camino Rocio (IRE) | 0.178 | 5.500 | 4.132 | True | -0.022 |
| Plumpton | 16:08 | Edelak (IRE) | 0.280 | 5.500 | 2.783 | True | 0.537 |
| Plumpton | 16:08 | Eternal Angel (FR) | 0.215 | 3.650 | 3.314 | True | -0.216 |
| Plumpton | 16:08 | Mojito des Mottes (FR) | 0.198 | 4.300 | 4.132 | True | -0.148 |
| Plumpton | 16:08 | Nap Hand (IRE) | 0.158 | 2.660 | 4.132 | False | -0.580 |
| Plumpton | 16:38 | Farhh Echo (IRE) | 0.233 | 2.800 | 3.314 | False | -0.349 |
| Plumpton | 16:38 | Khalk'eau Spigao (FR) | 0.200 | 6.600 | 4.132 | True | 0.319 |
| Plumpton | 17:08 | Shane's Spirit (IRE) | 0.522 | 1.860 | 1.598 | True | -0.029 |
| Plumpton | 17:08 | Kap In Hand (IRE) | 0.154 | 5.000 | 4.132 | True | -0.228 |