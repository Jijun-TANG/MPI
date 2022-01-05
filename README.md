## Using mpi4py to simulate life game with parallel process

### Propres libraries need to be installed for the program to work:
    numpy, mpi4py, tkinter

### When you try to launch the program, use
    mpiexec -n [NUMBER_OF_PROCESS] python version1.py [NAME_OF_INITIALISATIONS]

### Where initialisations are listed below:
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }

### NUMBER_OF_PROCESS-1 need to be divisible by the first size number after the initialisation name, for example:

    mpiexec -n 6 python version1.py blinker
    mpiexec -n 3 python version1.py toad
    mpiexec -n 4 python version1.py toad
    mpiexec -n 5 python version1.py acorn
    mpiexec -n 6 python version1.py acorn
    mpiexec -n 11 python version1.py acorn
    mpiexec -n 21 python version1.py acorn
    mpiexec -n 26 python version1.py acorn
    mpiexec -n 3 python version1.py beacon
    mpiexec -n 4 python version1.py beacon
    mpiexec -n 6 python version1.py boat
    mpiexec -n 6 python version1.py glider
    mpiexec -n 11 python version1.py glider
    mpiexec -n 21 python version1.py glider
    mpiexec -n 26 python version1.py glider
    mpiexec -n 6 python version1.py space_ship
    mpiexec -n 11 python version1.py die_hard
    mpiexec -n 18 python version1.py pulsar
    mpiexec -n 21 python version1.py u
    mpiexec -n 21 python version1.py flat
    
### Hope you a nice exploration of life game!
