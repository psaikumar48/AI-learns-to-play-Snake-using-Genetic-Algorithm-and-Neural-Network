import pygame
import os
import random
import numpy

# Input values
M=10                                         # number of blocks in horizental diriction
N=10                                         # number of blocks in verticle diriction
grid_size=25                                 # size of block
population_length=500         
parants_length=50
file_name= 'Snake game by GA.npz'            # file that saves the population after each generation is completed

# functions
def Snake_game():
    global Snake,screen,loop,mloop,pause_time
    pygame.init()
    screen = pygame.display.set_mode((M*grid_size,N*grid_size))
    (x1,y1)=random.choice(grids)
    (x2,y2)=random.choice([(x1,y1+1),(x1-1,y1),(x1,y1-1),(x1+1,y1)])
    Snake,steps,uniq,pause_time=[(x1,y1),(x2,y2)],0,[0]*(M*N-2),0
    food()
    loop=True
    while loop:
        steps=steps+1
        prediction_from_genetic_weights()
        update_snake()
        if len(Snake)==M*N:
            print('Great....Snake get maximum Score')
            loop=False
        elif snake_head==Food:
            food()
            Snake.append(snake_tail)
            pause_time =60  if len(Snake)==M*N-10 else pause_time
        elif snake_head not in grids or snake_head in snake_body:
            loop=False
        ev=pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                pygame.quit()
                mloop,loop=False,False
            elif event.type == pygame.KEYDOWN:
                pause_time = pause_time+key_sensitive  if event.key == pygame.K_UP else pause_time-key_sensitive if event.key == pygame.K_DOWN and pause_time>=key_sensitive else pause_time
        if (Snake[0],Food) not in uniq:
            uniq.append((Snake[0],Food))
            del uniq[0]
        else:
            loop=False
    score=len(Snake)-2
    return (score+0.5+0.5*(score-steps/(score+1))/(score+steps/(score+1)))*1000000,score,steps
def food():
    global Food
    snake_no_grids= [i for i in grids if i not in Snake]
    Food = random.choice(snake_no_grids)
def prediction_from_genetic_weights():
    global action
    lstop=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    lst=[(0,-1),(1,0),(0,1),(-1,0)]
    head_diriction=lstop[lst.index((Snake[0][0]-Snake[1][0],Snake[0][1]-Snake[1][1]))]
    (x,y)=Snake[0]
    d1=[(x,_) for _ in range(y-1,-1,-1)]
    d3=[(_,y) for _ in range(x+1,M)]
    d5=[(x,_) for _ in range(y+1,N)]
    d7=[(_,y) for _ in range(x-1,-1,-1)]
    d8=[(x-_,y-_) for _ in range(1,min(x,y)+1)]
    d4=[(x+_,y+_) for _ in range(1,min(M-x,N-y))]
    d2=[(x+_,y-_) for _ in range(1,max(M-x,N-y)) if (x+_,y-_) in grids]
    d6=[(x-_,y+_) for _ in range(1,max(M-x,N-y))  if (x-_,y+_) in grids]
    d,val=[d1,d2,d3,d4,d5,d6,d7,d8],min(M,N)-1
    wall_distance=[len(i)/val for i in d]
    food_presence=[(val-j.index(Food))/val if Food in j else 0 for j in d]
    body_presence=[min([dv.index(v) if v in Snake else val for v in dv])/val if dv else 0 for dv in d]
    vision=[j[i] for i in range(8) for j in [wall_distance,body_presence,food_presence]]
    input_layer=vision+head_diriction
    action=neural_network(input_layer)
def neural_network(ip):
    m1,s1=numpy.reshape(ip,(1,NN[0])),0
    for _ in range(len(AF)):
        l1,l2=NN[_],NN[_+1]
        s2,s3=s1+l1*l2,s1+l2+l1*l2
        m2,m3=numpy.reshape(weights[s1:s2],(l1,l2)),numpy.reshape(weights[s2:s3],(1,l2))
        m4=numpy.matmul(m1,m2)+m3
        m1,s1=AF[_](m4),s3
    return Actions[numpy.argmax(m1)]
def relu(x):
    lst=[_ if _>0 else 0 for _ in x[0]]
    return numpy.array(lst)
def sigmoid(x):
    lst=[1/(1+(numpy.exp(-1*_))) for _ in x[0]]
    return numpy.array(lst)
def update_snake():
    global snake_tail,snake_head,snake_body
    display()
    pygame.time.wait(pause_time)
    (x,y)=Snake[0]
    Snake.insert(0,(x+1,y)) if action == 'Right' else Snake.insert(0,(x-1,y)) if action == 'Left' else Snake.insert(0,(x,y+1)) if action == 'Bottum' else Snake.insert(0,(x,y-1))
    snake_tail=Snake.pop()
    snake_head,snake_body=Snake[0],Snake[1:]
def display():
    pygame.draw.rect(screen,(0,0,0), (0,0,M*grid_size,N*grid_size))
    pygame.draw.rect(screen,(255,255,255),(Snake[0][0]*grid_size,Snake[0][1]*grid_size,grid_size,grid_size))
    [pygame.draw.rect(screen,(255,255,255), (i[0]*grid_size,i[1]*grid_size,grid_size,grid_size),1) for i in Snake[1:]]
    pygame.draw.rect(screen,(0,255,0), (Food[0]*grid_size,Food[1]*grid_size,grid_size,grid_size))
    pygame.display.update()
def crossover():
    global offspring
    offspring=[]
    for _ in range(population_length-parants_length):
        parant1_id=random.choice(Roulette_wheel)
        parant2_id=random.choice(Roulette_wheel)
        while parant2_id==parant1_id: parant2_id=random.choice(Roulette_wheel)
        wts=[parants[parant1_id][i] if random.uniform(0, 1) < 0.5 else parants[parant2_id][i] for i in range(weights_length)]
        offspring.append(wts)
def mutation():
    global offspring
    for i in range(population_length-parants_length):
        for _ in range(int(weights_length*0.05)):
            plc=random.randint(0,weights_length-1)
            value=random.choice(numpy.arange(-0.5,0.5,step=0.001))
            offspring[i][plc]=offspring[i][plc]+value

# Main algorthim
NN=[28,8,4]                                  # Neurol network layers
AF=[relu,sigmoid]                            # Activation functions 
pause_time,key_sensitive,generation_length,mloop=0,15,2000,True
grids=[(i,j) for i in range(M) for j in range(N)]
Actions=['Top','Right','Bottum','Left']
Roulette_wheel=list(range(0,int(0.2*parants_length)))*3+list(range(int(0.2*parants_length),int(0.5*parants_length)))*2+list(range(int(0.5*parants_length),parants_length))
weights_length=sum([NN[_]*NN[_+1]+NN[_+1] for _ in range(len(NN)-1)])
if file_name not in os.listdir(os.getcwd()):
    population,statis=numpy.random.choice(numpy.arange(-1,1,step=0.001),size=(population_length,weights_length),replace=True),numpy.array([0,0,0,0])
    Generation,High_score=1,0
else:
    IP=numpy.load(file_name)
    population,statis=IP['POPULATION'],IP['STATIS']
    Generation,High_score=statis[-1][0]+1,statis[-1][-1]


while Generation<=generation_length and mloop:
    print('###################### ','Generation ',Generation,' ######################')
    Fitness,Score,i=[],[],0
    while i<population_length and mloop:
        weights,i=list(population[i,:]),i+1
        fittness,score,steps=Snake_game()
        print('Chromosome ',"{:03d}".format(i),' >>> ','Score : ',"{:03d}".format(score),', Steps : ',"{:04d}".format(steps),', Fittness : ',fittness)
        Fitness.append(fittness),Score.append(score)
    parants,max_fittness,avg_score,j=[],max(Fitness),sum(Score)/len(Score),0
    while j<parants_length and mloop:
        j,parant_id=j+1,Fitness.index(max(Fitness))
        Fitness[parant_id]=-999
        parants.append(list(population[parant_id,:]))
    while mloop and j==parants_length:
        j=j+1
        High_score=max(Score) if max(Score) > High_score else High_score
        print('Generation high score : ',max(Score),', Generation Avg score : ',avg_score,', Overall high score : ',High_score)
        crossover()
        mutation()
        statis=numpy.row_stack((statis,numpy.array([Generation,max(Score),avg_score,High_score])))
        population=numpy.reshape(parants+offspring,(population_length,-1))
    Generation=Generation+1
pygame.quit()
numpy.savez(file_name,POPULATION=population,STATIS=statis)
