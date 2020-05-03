import pygame
import random
from scipy.spatial import distance
import numpy as np

# Snake game functions(food,display,update_snake,snake_game)
def food():
    global Food
    snake_no_grids= [i for i in grids if i not in Snake]
    Food = random.choice(snake_no_grids)
def display():
    pygame.draw.rect(screen,(0,0,0), (0,0,M*grid_size,N*grid_size))
    for i in Snake:
        pygame.draw.rect(screen,(255,255,255), (i[0]*grid_size,i[1]*grid_size,grid_size,grid_size),1)
    pygame.draw.rect(screen,(255,255,255), (Food[0]*grid_size,Food[1]*grid_size,grid_size,grid_size))
    pygame.display.update()
def update_snake():
    global snake_tail,snake_head,snake_body
    (x,y)=Snake[0]
    if action == 'Right' :
        Snake.insert(0,(x+1,y))
    elif action == 'Left' :
        Snake.insert(0,(x-1,y))
    elif action == 'Top' :
        Snake.insert(0,(x,y-1))
    elif action == 'Bottum' :
        Snake.insert(0,(x,y+1))
    snake_tail=Snake.pop()
    snake_head=Snake[0]
    snake_body=Snake[1:len(Snake)]
    display()
def snake_game(steps,weights):
    global Snake,screen
    Snake_wait_time,overall_score,step_count=0,0,0
    mloop=True
    while mloop:
        pygame.init()
        screen = pygame.display.set_mode((M*grid_size,N*grid_size))  
        (x1,y1)=random.choice(grids)
        (x2,y2)=random.choice([(x1,y1+1),(x1-1,y1),(x1,y1-1),(x1+1,y1)])
        Snake=[(x1,y1),(x2,y2)]
        food()
        loop=True
        while loop:
            step_count+=1
            pygame.time.wait(Snake_wait_time)
            finding_output_from_weights_ipnurons(weights)
            update_snake()
            if snake_head==Food:
                overall_score+=1
                food()
                Snake.append(snake_tail)
            elif snake_head not in grids or snake_head in snake_body or (step_count>int(steps/5) and overall_score<=0):
                overall_score+=-1
                loop=False
            ev=pygame.event.get()
            for event in ev:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    loop=False
                    mloop=False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        loop=False
                    elif event.key == pygame.K_UP:
                        Snake_wait_time+=10            
                    elif event.key == pygame.K_DOWN:
                        if Snake_wait_time >=10:
                            Snake_wait_time-=10
            if step_count>=steps:
                loop=False
                mloop=False
    return overall_score   

# functions used to find the inputs required for neural network(direction_view,train_data)
def direction_view(p,ip,food_count=0,body_count=0):
        (x1,y1)=operation[p]
        x,y=ip[0]+x1,ip[1]+y1
        if 0<=x<M and 0<=y<N:
            if (x,y)==Food:
                food_count+=1
            if (x,y) in Snake[1:]:
                body_count+=1
            return direction_view(p,(x,y),food_count,body_count)
        else:
            if body_count>0:
                body_count=1
            wall_distance=int(distance.euclidean(Snake[0],ip))
            return [wall_distance,food_count,body_count]
def train_data():
    global operation
    operation=[(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
    op,z=[],[0]*8
    for i in range(8):
        op.extend(direction_view(i,Snake[0]))
    (x1,y1),(x2,y2),(x3,y3),(x4,y4)=Snake[0],Snake[1],Snake[-2],Snake[-1]
    idx1=[(x1,y1+1),(x1-1,y1),(x1,y1-1),(x1+1,y1)].index((x2,y2))
    idx2=4+[(x3,y3+1),(x3-1,y3),(x3,y3-1),(x3+1,y3)].index((x4,y4))
    z[idx1],z[idx2]=1,1
    op.extend(z)
    return op
# function used to calculate value of output neurons based on given weights(finding_output_from_weights_ipnurons)
def finding_output_from_weights_ipnurons(weights):
    global len_weights
    input=weights+train_data()
    global action
    def softmax(z):
        s = np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1, 1)
        return s
    len_input_neurons=32
    len_hidden_layer=8
    len_output_neurons=4
    len_weights=(len_input_neurons*len_hidden_layer)+(len_hidden_layer*len_output_neurons)
    W1=np.reshape(input[:256],(8,32))
    W2=np.reshape(input[256:len_weights],(4,8))
    IP=np.reshape(input[len_weights:],(32,1))
    Z1 = np.matmul(W1,IP)
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1)
    A2 = softmax(Z2)
    action=Actions[np.argmax(A2)]

# function applied on Childrens (crossover,mutation
def crossover():
    offspring=[]
    for _ in range(1000):
        if len(offspring)<population_size-parants_size:
            parant1_id=random.randint(0,parants_size-1)
            parant2_id=random.randint(0,parants_size-1)
            wts=[]
            for i in range(len_weights):
                if random.uniform(0, 1) < 0.5:
                    wts.append(parants[parant1_id][i])
                else:
                    wts.append(parants[parant2_id][i])
            offspring.append(wts)
        else:
            break
    return offspring
def mutation(offspring):
    for i in range(population_size-parants_size):
        for _ in range(35):
            plc=random.randint(0,len_weights-1)
            value=random.choice(np.arange(-1,1,step=0.001))
            offspring[i][plc]=offspring[i][plc]+value
    return offspring