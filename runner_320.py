
# frqi circuit from https://github.com/Shedka/citiesatnight


import utils
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
from qiskit import execute, Aer, BasicAer
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
import matplotlib.pyplot as plt
from resizeimage import resizeimage
from PIL import Image, ImageOps
import frqi
# import quantum_edge_detection as qed



# Insert API key generated after registring in IBM Quantum Experience
# IBMQ.save_account('API KEY')

IBMQ.load_account()
provider = IBMQ.get_provider( group='open', project='main')


# dimensions of the image
size=32

#target image
images=utils.get_Cat_320_image()

# New blank image
new_im = Image.new('RGB', (320, 320))

for k in range(10):
    for j in range(10):
        normalized_image=utils.large_image_normalization(images,32+(32*k),32+(32*j))
        genimg= np.array([])
        anc = QuantumRegister(1, "anc")
        img = QuantumRegister(11, "img")
        anc2 = QuantumRegister(1, "anc2")
        c = ClassicalRegister(12)
        qc = QuantumCircuit(anc, img, anc2, c)

        for i in range(1, len(img)):
                qc.h(img[i])


        for i in range(len(normalized_image)):
                if normalized_image[i] != 0:
                        frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[j] for j in range(1,len(img))])
        qc.measure(img, c[1:12])
        print(qc.depth())
        numOfShots = 1000000
        result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
        for i in range(len(normalized_image)):
                try:
                        genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
                except KeyError:
                        genimg = np.append(genimg,[0.0])

        genimg *= 32.0 * 255.0 
        genimg = genimg.astype('int')
        genimg = genimg.reshape((32,32))
        im=Image.fromarray(genimg)

        new_im.paste(im,(32*k,32*j))
new_im.show()
new_im.save('Result_320.png')

