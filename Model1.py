import pygame
import os
import random
import numpy

def food():
    global Food
    snake_no_grids= [i for i in grids if i not in Snake]
    Food = random.choice(snake_no_grids)
def display():
    pygame.draw.rect(screen,(0,0,0), (0,0,M*grid_size,N*grid_size))
    pygame.draw.rect(screen,(255,255,255),(Snake[0][0]*grid_size,Snake[0][1]*grid_size,grid_size,grid_size))
    [pygame.draw.rect(screen,(255,255,255), (i[0]*grid_size,i[1]*grid_size,grid_size,grid_size),1) for i in Snake[1:]]
    pygame.draw.rect(screen,(0,255,0), (Food[0]*grid_size,Food[1]*grid_size,grid_size,grid_size))
    pygame.display.update()
def update_snake():
    global snake_tail,snake_head,snake_body
    display()
    (x,y)=Snake[0]
    Snake.insert(0,(x+1,y)) if action == 'Right' else Snake.insert(0,(x-1,y)) if action == 'Left' else Snake.insert(0,(x,y+1)) if action == 'Bottum' else Snake.insert(0,(x,y-1))
    snake_tail=Snake.pop()
    snake_head,snake_body=Snake[0],Snake[1:]
def prediction_from_genetic_weights():
    global action
    lstop=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    lst=[(0,-1),(1,0),(0,1),(-1,0)]
    head_diriction=lstop[lst.index((Snake[0][0]-Snake[1][0],Snake[0][1]-Snake[1][1]))]
    tail_diriction=lstop[lst.index((Snake[-2][0]-Snake[-1][0],Snake[-2][1]-Snake[-1][1]))]
    (x,y)=Snake[0]
    d1=[(x,_) for _ in range(y-1,-1,-1)]
    d3=[(_,y) for _ in range(x+1,M)]
    d5=[(x,_) for _ in range(y+1,N)]
    d7=[(_,y) for _ in range(x-1,-1,-1)]
    d8=[(x-_,y-_) for _ in range(1,min(x,y)+1)]
    d4=[(x+_,y+_) for _ in range(1,min(M-x,N-y))]
    d2=[(x+_,y-_) for _ in range(1,max(M-x,N-y)) if (x+_,y-_) in grids]
    d6=[(x-_,y+_) for _ in range(1,max(M-x,N-y))  if (x-_,y+_) in grids]
    d=[d1,d2,d3,d4,d5,d6,d7,d8]
    wall_distance=[len(i)/9 for i in d]
    food_presence=[(9-j.index(Food))/9 if Food in j else 0 for j in d]
    body_presence=[min([dv.index(v) if v in Snake else 9 for v in dv])/9 if dv else 0 for dv in d]
    vision=[j[i] for i in range(8) for j in [wall_distance,body_presence,food_presence]]
    input_layer=vision+head_diriction+tail_diriction
    hidden_l1=numpy.matmul(numpy.reshape(input_layer,(1,32)),numpy.reshape(weights[:640],(32,20)))+numpy.reshape(weights[640:660],(1,20))
    opt_hidden_l1=relu(hidden_l1)
    hidden_l2=numpy.matmul(opt_hidden_l1,numpy.reshape(weights[660:900],(20,12)))+numpy.reshape(weights[900:912],(1,12))
    opt_hidden_l2=relu(hidden_l2)
    hidden_l3=numpy.matmul(opt_hidden_l2,numpy.reshape(weights[912:960],(12,4)))+numpy.reshape(weights[960:],(1,4))
    opt_hidden_l3=sigmoid(hidden_l3)
    action=Actions[numpy.argmax(opt_hidden_l3)]
def relu(x):
    lst=[_ if _>0 else 0 for _ in x[0]]
    return numpy.array(lst)
def sigmoid(x):
    lst=[1/(1+(numpy.exp(-1*_))) for _ in x[0]]
    return numpy.array(lst)
def Snake_game():
    global Snake,screen,loop,mloop,pause_time,High_score
    pygame.init()
    screen = pygame.display.set_mode((M*grid_size,N*grid_size))
    (x1,y1)=random.choice(grids)
    (x2,y2)=random.choice([(x1,y1+1),(x1-1,y1),(x1,y1-1),(x1+1,y1)])
    Snake,steps,uniq=[(x1,y1),(x2,y2)],1,[0]*81
    food()
    loop=True
    while loop:
        blocks=[1 if (i not in grids or i in Snake) else 0 for i in [(Snake[0][0],Snake[0][1]-1),(Snake[0][0]+1,Snake[0][1]),(Snake[0][0],Snake[0][1]+1),(Snake[0][0]-1,Snake[0][1])]]
        steps=steps+1 if sum(blocks)==2 else steps
        pygame.time.wait(pause_time)
        prediction_from_genetic_weights()
        update_snake()
        if snake_head==Food:
            food()
            Snake.append(snake_tail)
        elif snake_head not in grids or snake_head in snake_body:
            loop=False
        ev=pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                pygame.quit()
                mloop,loop=False,False
            elif event.type == pygame.KEYDOWN:
                pause_time = pause_time+25  if event.key == pygame.K_UP else pause_time-25 if event.key == pygame.K_DOWN and pause_time>=25 else pause_time
                if event.key == pygame.K_ESCAPE:
                    loop,steps = False,82
        if (Snake[0],Food) not in uniq:
            uniq.append((Snake[0],Food))
            del uniq[0]
        else:
            loop=False
    score=len(Snake)-1
    High_score=score-1 if score-1 > High_score else High_score
    return steps**((score**2.2)/(steps)),score-1

def crossover():
    global offspring
    offspring=[]
    for _ in range(population_length-parants_length):
        parant1_id=random.randint(0,84)
        parant2_id=random.randint(0,84)
        wts=[parants_selection[parant1_id][i] if random.uniform(0, 1) < 0.5 else parants_selection[parant2_id][i] for i in range(964)]
        offspring.append(wts)
def mutation():
    global offspring
    for i in range(population_length-parants_length):
        for _ in range(35):
            plc=random.randint(0,963)
            value=random.choice(numpy.arange(-0.5,0.5,step=0.001))
            offspring[i][plc]=offspring[i][plc]+value

M,N,grid_size=10,10,20
grids=[(i,j) for i in range(M) for j in range(N)]
Actions=['Top','Right','Bottum','Left']
pause_time,generation_length,population_length,parants_length=0,2000,500,50
if 'population32_20_12_4_nn34.npy' not in os.listdir(os.getcwd()):
    population=numpy.random.choice(numpy.arange(-1,1,step=0.001),size=(population_length,964),replace=True)
    Generation,High_score=0,0
else:
    population=numpy.load('population32_20_12_4_nn34.npy')
    Generation,High_score=population[population_length][0],population[population_length][1]

mloop=True
for i in range(generation_length+1):
    if  mloop:
        Number_of_generations=Generation+i+1
        print('########### ','Generation ',Number_of_generations,' ###########')
        Fitness,gen_high_score=[],0
        for j in range(population_length):
            if  mloop:
                weights=list(population[j,:])
                fittness,score=Snake_game()
                print('Chromosome ',"{:02d}".format(j),' >>> ','Score : ',"{:02d}".format(score),', Fittness : ',fittness)
                Fitness.append(fittness)
                gen_high_score=score if score > gen_high_score else gen_high_score
        if  mloop:
            print('Generation high score : ',gen_high_score,', Generation high fittness : ',max(Fitness),', Overall high score : ',High_score,)
            parants,parants_selection=[],[]
            for k in range(parants_length):
                parant_id=Fitness.index(max(Fitness))
                Fitness[parant_id]=-999
                parants.append(list(population[parant_id,:]))
                if k<10:
                    parants_selection.append(list(population[parant_id,:]))
                    parants_selection.append(list(population[parant_id,:]))
                    parants_selection.append(list(population[parant_id,:]))
                elif k<25:
                    parants_selection.append(list(population[parant_id,:]))
                    parants_selection.append(list(population[parant_id,:]))
                else:
                    parants_selection.append(list(population[parant_id,:]))
            crossover()
            mutation()
            population=numpy.reshape(parants+offspring,(population_length,-1))

pygame.quit()
Number_of_generations=Number_of_generations if mloop else Number_of_generations-1
info=numpy.array([Number_of_generations,High_score]+[0]*962)
population=numpy.insert(population,population_length,info,axis=0)
numpy.save('population32_20_12_4_nn34',population)      
    
