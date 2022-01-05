import numpy as np
import sys
import functools
import operator
from lifegame import grille,App
 

"""
L'astuce de commande:
le nombre de (processus-1) doivent être divisible par dim[0]

par exemple:

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
"""

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


if len(sys.argv)>1:
    choice = sys.argv[1]
else:
    choice='space_ship'
dim = dico_patterns[choice][0]



# Importation du module MPI
from mpi4py import MPI
comm_couple = MPI.COMM_WORLD.Dup()
rank_couple = comm_couple.rank
size_couple = comm_couple.size

nombre_cells_glob = dim[0]*dim[1]
nombre_ligne_mat_loc = dim[0]//(np.sqrt(size_couple-1).astype(int))
nombre_col_mat_loc = dim[1]//(np.sqrt(size_couple-1).astype(int))

# Ouverture d'un fichier nom unique mode simple
#fileName = f"sortie{rank_couple:03d}.txt"
#file = open(fileName, mode="w")

# séparation entre le processus qui affiche et les processus qui calculent
global_grille=None

if rank_couple == 0:
    status_couple = 0
    
    init_pattern = dico_patterns[choice]
    global_grille=grille(*init_pattern)
    
    global_grid = global_grille.cells
    appli = App((800,800),global_grille)

    print("check initialisation")
    print(global_grid)
    print("initialisation complet")
    #si la nbr de processus est 1 on fait un program sequentiel
    if size_couple==1:
        print("Le nombre de processus total est 1, le program s'execute dans un seul processus")
        while True:
            diff = global_grille.compute_next_iteration()
            appli.draw(diff)
else :
    status_couple = 1
comm = comm_couple.Split(status_couple,rank_couple)
rank = comm.Get_rank()
size = comm.Get_size()

if rank_couple==0:
    size=-1
    comm_couple.send(global_grille,dest=1,tag=0)
elif rank_couple==1:
    global_grille=comm_couple.recv(source=0,tag=0)
#liste de nombre de processus qui peut potentiellement former des sous matrices en carré au cours de calcul parallèle,
#si le nombre de processus donné n'est pas dans cette liste, on fait un algo universel pour n'importe quelle nombre de processus
#ici on traite les distributions importantes au dessous qui peuvent potentiellement convenir à notre taille du problème



if size==1: #si le nbr de processus en travail est 1 on fait un program sequentiel
    if rank_couple==0:
        print("Le nombre de work processus est 1, le program s'execute dans un seul processus")
        while True:
            diff = global_grille.compute_next_iteration()
            appli.draw(diff)
else:#si le nbr de processus est insuffisant pour diviser la grilles en block, on fait une division par ligne, qui est aussi universelle pour n'importe quelle nombre de processus
    nb_it=0
    while True:
        if status_couple==1:
            sendbuff=None
            recvbuf=None
            residu_a_fin=False
            if dim[0]%size==0:
                longeur=int(dim[0]/size)
            else:
                residu_a_fin=True
                longeur=dim[0]//size
            nom_elements_par_processus=longeur*dim[1]
            if residu_a_fin:#si le nombre de processus n'est pas divisible par le nombre de rang on rajoute un rang à la dernière bloc de matrice
                if rank==size-1:
                    recvbuf=np.empty((2,nom_elements_par_processus),dtype=np.uint8)
                else:
                    recvbuf=np.empty((1,nom_elements_par_processus),dtype=np.uint8)
            else:
                recvbuf=np.empty((1,nom_elements_par_processus),dtype=np.uint8)

            """
            Rapport: cette méthode pour traiter les cas où le nombre de processus n'est pas divisible par le rang n'a pas réussi. La fonction Scatterv distribue les elements d'une façon équilibre à
            chaque processus. Donc si le nombre de processus n'est pas divisble par le dim[0], ça génère encore des erreurs
            """
            
            if rank==0:
                sendbuff=global_grille.cells
                #print("grille taille", global_grille.dimensions)
                sendbuff=sendbuff.flatten()
                #print("sendbuff: ", sendbuff.shape)
                #print(sendbuff)
            #print("recvbuf: ", recvbuf.shape)
            #print(recvbuf)
            try:
                comm.Scatterv(sendbuff,recvbuf,root=0)
            except:
                print("At ", rank, "problem with comm.Scatterv")
                exit()
            vector_up=np.empty((1,dim[1]),dtype=np.uint8)
            vector_down=np.empty((1,dim[1]),dtype=np.uint8)
            vu=None
            vd=None

            """
            La matrices est coupée par ligne, chaque processus prend en charge des lignes identiques et envoyent ses bords supérieur et inférieur aux processus sup et inf
            """
            if residu_a_fin and rank==size-1:
                recvbuf=recvbuf.reshape((2*longeur,dim[1]))
            else:
                recvbuf=recvbuf.reshape((longeur,dim[1]))
            if rank==0:
                vector_up=recvbuf[0]
                vector_down=recvbuf[recvbuf.shape[0]-1]
                vu=np.empty(vector_up.shape,dtype=np.uint8)
                vd=np.empty(vector_down.shape,dtype=np.uint8)
                comm.Send(vector_up,dest=size-1,tag=0)
                comm.Send(vector_down,dest=1,tag=1)
                comm.Recv(vu,source=size-1,tag=2*size-1)
                comm.Recv(vd,source=1,tag=2)
            elif rank==size-1:
                vector_up=recvbuf[0]
                vector_down=recvbuf[recvbuf.shape[0]-1]
                vu=np.empty(vector_up.shape,dtype=np.uint8)
                vd=np.empty(vector_down.shape,dtype=np.uint8)
                comm.Send(vector_up,dest=rank-1,tag=2*rank)
                comm.Send(vector_down,dest=0,tag=2*rank+1)
                comm.Recv(vu,source=rank-1,tag=2*(rank-1)+1)
                comm.Recv(vd,source=0,tag=0)
            else:
                vector_up=recvbuf[0]
                vector_down=recvbuf[recvbuf.shape[0]-1]
                vu=np.empty(vector_up.shape,dtype=np.uint8)
                vd=np.empty(vector_down.shape,dtype=np.uint8)
                comm.Send(vector_up,dest=rank-1,tag=2*rank)
                comm.Send(vector_down,dest=rank+1,tag=2*rank+1)
                comm.Recv(vu,source=rank-1,tag=2*(rank-1)+1)
                comm.Recv(vd,source=rank+1,tag=2*rank+2)
            """
            #les prints pour faire des débougage
            print('\n')
            print("------iteration------\n",nb_it)
            print("rank")
            print(rank)
            print("local matrix")
            print(recvbuf)
            print("upper vector")
            print(vu)
            print("lower vector")
            print(vd)
            print('\n')
            """
            next_cells=recvbuf.copy()
            if residu_a_fin and rank==size-1:
                recvbuf=np.vstack((vu,recvbuf))
                recvbuf[longeur]=vd
            elif not residu_a_fin:
                recvbuf=np.vstack((vu,recvbuf))
                recvbuf=np.vstack((recvbuf,vd))
            diff_loc=[]#difference local des coordonnées globales
            def coordinates_gl(i,j):
                return nom_elements_par_processus*rank+i*dim[1]+j+1
            ny = recvbuf.shape[0]
            nx = recvbuf.shape[1]

            for i in range(1,longeur+1):
                i_above = i-1
                i_below = i+1 
                for j in range(dim[1]):
                    j_left = (j-1+nx)%nx
                    j_right= (j+1)%nx
                    voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                    voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                    voisines = np.array(recvbuf[voisins_i,voisins_j])
                    nb_voisines_vivantes = np.sum(voisines)
                    if recvbuf[i,j] == 1: # Si la cellule est vivante
                        if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                            next_cells[i-1,j-1] = 0 # Cas de sous ou sur population, la cellule meurt
                            ig=coordinates_gl(i-1,j-1)
                            diff_loc.append(ig)
                        else:
                            next_cells[i-1,j-1] = 1 # Sinon elle reste vivante
                    elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                        next_cells[i-1,j-1] = 1         # Naissance de la cellule
                        ig=coordinates_gl(i-1,j-1)
                        diff_loc.append(ig)
                    else:
                        next_cells[i-1,j-1] = 0
            diff_glob=comm.gather(diff_loc,root=0)
            if rank==0:
                diff_glob=functools.reduce(operator.iconcat,diff_glob,[])
                comm_couple.send(diff_glob,dest=0,tag=13000)

                """
                debugging print
                print(nb_it)
                print('\n')
                print(diff_glob)
                """
        if status_couple==0:
            diff_glob=comm_couple.recv(source=1,tag=13000)
            global_grid=global_grille.cells.copy()
            glob_grid_shape=global_grid.shape
            global_grid=global_grid.flatten()
            for ind in diff_glob:
                if global_grid[ind]==1:
                    global_grid[ind]=0
                else:
                    global_grid[ind]=1
            global_grille.cells=global_grid.reshape(glob_grid_shape)
            comm_couple.send(global_grille,dest=1,tag=14000)
            """
            #debugging print
            print("~~~~~~~~~~~~~GLOBAL GRID~~~~~~~~~~~~~~~~\n")
            print(nb_it)
            print(global_grille.cells)
            """
            appli.draw(diff_glob)
        if rank_couple==1:
            global_grille=comm_couple.recv(source=0,tag=14000)
        nb_it+=1













