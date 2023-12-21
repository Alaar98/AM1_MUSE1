from numpy import array, zeros, linspace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ODE_Arenstorf import Arenstorf
from Cauchy_problem import Cauchy_problem_RK4_emb
from Temporal_schemes import  adaptive_RK_emb

#----------------------------------------------------------
#Initial Conditions
U0 = array([0.994, 0, 0, -2.00158510637908252240537862224])

#Parameters definition
T = 17.0652165601579625588917206249
N = 100000
dt = T/N
t = linspace(0,T,N)
x = array(zeros(N))
y = array(zeros(N))
x[0] = U0[0]
y[0] = U0[1]

scheme = adaptive_RK_emb
sol,h = Cauchy_problem_RK4_emb (Arenstorf, t, U0, scheme)


def create_animation_stepsize(h,t, anim_fps, anim_duration, anim_save_name,N):
   # Funciones para crear la animacion:
   # - Función de inicializacion: se llama para crear la trama base vacia
   def init():
      line.set_data([], [])
      return line,
   # - Funcion de actualizacion para cada cuadro de la animacion
   def update(frame):
      x1 = t[:frame]  # Obtener datos hasta el cuadro actual
      y1 = h[:frame]
      line.set_data(x1, y1)
      return line,
 
   # Crear la figura usando plt.figure
   plt.figure(figsize=(8, 8))
   plt.tick_params(labelsize=13)
 
   # Crear una linea vacia que se actualizara en la animacion
   line, =  plt.plot([], [], '-',color='red')
 
   # Definir los parámetros de la animación
   anim_frames = linspace(0, N, int(anim_fps*anim_duration+1), dtype=int)
   anim_interval = (anim_duration / int(anim_fps*anim_duration+1)) * 1000
 
   # Definir los límites de los ejes de la grafica
   rangex = (max(t[:]) - min(t[:])) * 0.1
   rangey = (max(h[:]) - min(h[:])) * 0.1
   xlim = [min(t[:]) - rangex, max(t[:]) + rangex]
   ylim = [min(h[:]) - rangey, max(h[:]) + rangey]
 
   # Definicion de propiedades de la grafica
   plt.xlim(xlim)
   plt.ylim(ylim)
   plt.xlabel('Time', fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'k'})
   plt.ylabel('Step Size', fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'k'})
   plt.title('Adaptive Step Size over Time', fontdict = {'fontsize':18, 'fontweight':'normal', 'color':'k'})
   # Creacion y guardado de la animacion
   ani = FuncAnimation(plt.gcf(), update, frames=anim_frames, init_func=init, blit=True, interval=anim_interval)
   ani.save(anim_save_name, writer='Pillow', fps=anim_fps)
 
   plt.show()


# Crear la animacion
create_animation_stepsize(h,t,60,4,'Sepsize_Variation.gif',N)