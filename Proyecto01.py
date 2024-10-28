"""
Trajectory generation based on b-spline curves and Coppeliasim Pioneer

Author: Jose Julian Sierra Alvarez
University of Guanajuato (2024)
"""

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import time
import math as m

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def v2u(v, omega, r, L):
    ur = v/r + L*omega/(2*r)
    ul = v/r - L*omega/(2*r)
    return ur, ul

def angdiff(t1, t2):
    """
    Compute the angle difference, t2-t1, restricting the result to the [-pi,pi] range
    """
    # The angle magnitude comes from the dot product of two vectors
    angmag = m.acos(m.cos(t1)*m.cos(t2)+m.sin(t1)*m.sin(t2))
    # The direction of rotation comes from the sign of the cross product of two vectors
    angdir = m.cos(t1)*m.sin(t2)-m.sin(t1)*m.cos(t2)
    return m.copysign(angmag, angdir)

print('Program started')

# Inicializacion
client = RemoteAPIClient()
sim = client.getObject('sim')

motorL = sim.getObject("/PioneerP3DX/leftMotor")
motorR = sim.getObject("/PioneerP3DX/rightMotor")
robot = sim.getObject("/PioneerP3DX")

timTotal = 150 # Tiempo en segundos
Kv = 0.25 # Que tan rapido va hacia adelante
Kh = 0.45 # Que tan rapido gira
r = 0.5*0.195
L = 0.311
turn_time = 2 # tiempo que tomara para evitar objetos
turn_speed = 2.0 * m.pi / 10.0 # definimos el parametro de giro para evadir 
		

sens1 = sim.getObject("/PioneerP3DX/ultrasonicSensor[3]") 
sens2 = sim.getObject("/PioneerP3DX/ultrasonicSensor[4]") # Inicializacion de los sensores

# Puntos a recorrer:
'''
# arreglo que inicia en 0 con 4 puntos aleatorios a visitar desde -10 a 10
xarr = np.concatenate((np.array([0]),np.random.randint(-10,11, size = 4)))
yarr = np.concatenate((np.array([0]),np.random.randint(-10,11, size = 4)))
'''
# arreglo que inicia en 0 con 9 puntos aleatorios a visitar desde -5 a 5
xarr = np.concatenate((np.array([0]),np.random.randint(-1,2, size = 1),np.random.randint(-5,6, size = 8)))
yarr = np.concatenate((np.array([0]),np.random.randint(-1,2, size = 1),np.random.randint(-5,6, size = 8)))

puntos = xarr.shape[0] - 1

print('Puntos que debe cruzar: x = ',xarr, ' y = ', yarr)
tarr = np.linspace(0, timTotal, xarr.shape[0])

xc = spi.splrep(tarr, xarr, s = 0, k = min(puntos, 5)) # objeto interpolador para evaluar
yc = spi.splrep(tarr, yarr, s = 0, k = min(puntos, 5)) # objeto interpolador para evaluar

sim.startSimulation()
# Deteccion de la posicion del robot
xpos = []
ypos = []
# Medicion del interpolador
xideal = []
yideal = []
# Deteccion de objetos en ruta
xObj = np.empty(0)
yObj = np.empty(0)
pos = sim.getObjectPosition(robot, -1)
print(f'Posicion inicial robot {pos}')

timInicio = sim.getSimulationTime()
# En lugar de medir el error medir el tiempo que falta para llegar.
while (sim.getSimulationTime() - timInicio) <= timTotal:
	# Obtencion de la medicion de los sensores frontales
	res1, dist1, _, _, _ = sim.readProximitySensor(sens1)
	res2, dist2, _, _, _ = sim.readProximitySensor(sens2)

	carpos = sim.getObjectPosition(robot, -1) # Posicion x y z del carro en el plano, nos interesa solo x y
	carrot = sim.getObjectOrientation(robot, -1) # Direcciones de x y z, car rotation, donde solo queremos la de z
	
	tcurr = sim.getSimulationTime() - timInicio # Medimos la cantidad de tiempo que ha pasado
	xd = spi.splev(tcurr, xc, der = 0)
	yd = spi.splev(tcurr, yc, der = 0)
	xideal.append(xd)
	yideal.append(yd)
		
	errp = m.sqrt((xd - carpos[0]) ** 2 + (yd - carpos[1]) ** 2)
	angd = m.atan2(yd - carpos[1], xd - carpos[0]) # tan-¹((y^d-y)/(x^d-x)) para obtener el angulo deseado
	errh = angdiff(carrot[2], angd) # Error angular, el signo indica para donde girar
	
	# Ajustamos dinamicamente la velocidad del robot acorde a los errores
	#print (f"errp = {errp} errh = {errh}")
	if errp > 3 and errp < 5:
		Kv = min (0.85, Kv + 0.01)  # Incrementara levemente la velocidad si el error es muy grande hasta un tope de velocidad
	else:
		Kv = max(0.25, Kv - 0.05)  # Decrementara la velocidad conforme llegue al error sin evitar que se detenga
		# La velocidad no puede ser menor a 0.5
	if abs(errh) > 1 and abs(errh) <= 1.7:
		#Kh += 0.01  # Incrementara levemente la velocidad angular si el error es muy grande
		Kh = min (1.0, Kh + 0.01)
	elif abs(errh) > 1.7 and abs(errh) < 2:
		Kh = min (1.0, Kh + 0.05)
		Kv = 0.25 # Reducimos toda la velocidad para que gire mejor
	else:
		Kh = max(0.45, Kh - 0.05)  # Decrementara la velocidad angular conforme llegue al error sin evitar que se detenga
		# La velocidad angular no puede ser menor a 0.3

	# Si se detectan objetos en un sensor:
	if res1 and res2:
		avg_dist = (dist1 + dist2) / 2
	elif res1:
		avg_dist = dist1
	elif res2:
		avg_dist = dist2
	else:
		avg_dist = None # Promediamos las mediciones de los sensores o solo capturamos el izquierdo o derecho

	if avg_dist is not None and avg_dist < 0.9: # Deteccion de objetos con el promedio de las mediciones
		print(f"Objeto detectado a {avg_dist} metros frente al Robot. Activando protocolo de evasión...")
		pos = sim.getObjectPosition(robot, -1)
		xObj = np.append(xObj, pos[0])
		yObj = np.append(yObj, pos[1])
		Kh = Kh / 2 
		Kv = Kv / 2 # reducimos velocidades para maniobrar adecuadamente
		sim.setJointTargetVelocity(motorL, 0)
		sim.setJointTargetVelocity(motorR, 0)
		time.sleep(1)
		ur, ul = v2u(0, turn_speed, r, L) # definimos velocidades para evasion
		sim.setJointTargetVelocity(motorL, ul)
		sim.setJointTargetVelocity(motorR, ur)
		time.sleep(turn_time)
		sim.setJointTargetVelocity(motorL, Kv / r) # restauramos parcialmente la velocidad y movemos el robot
		sim.setJointTargetVelocity(motorR, Kv / r)
		Kh = Kh * 2
		Kv = Kv * 2 # restauramos velocidades para que siga con la ruta
	else:
		sim.setJointTargetVelocity(motorL, Kv / r)
		sim.setJointTargetVelocity(motorR, Kv / r)
	v = Kv * errp # aplicamos el control de velocidad.
	omega = Kh * errh # aplicamos el control de velocidad angular.
	
	ur, ul = v2u(v, omega, r, L)
	sim.setJointTargetVelocity(motorL, ul)
	sim.setJointTargetVelocity(motorR, ur)
	pos = sim.getObjectPosition(robot, -1)
	xpos.append(pos[0])
	ypos.append(pos[1])

pos = sim.getObjectPosition(robot, -1)
print(f'Posicion final robot {pos}')
sim.stopSimulation()

# Graficamos las posiciones
plt.plot(xpos, ypos, marker='o', color = 'green', label = "Ruta real")
plt.plot(xideal, yideal, marker='o', color='red', label = "Ruta ideal")
plt.scatter(xarr, yarr, marker = 's', color = 'magenta', label = "Puntos a cruzar", zorder=10)
plt.scatter(xObj, yObj, marker = 's', color = 'blue', label = "Deteccion de objeto", zorder=5)
plt.xlabel('Posición X (m)')
plt.ylabel('Posición Y (m)')
plt.title('Trayectoria del Robot Pioneer P3DX')
plt.grid(True)
plt.legend()
plt.show()
