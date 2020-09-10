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
import quantum_edge_detection as qed
# import quantum_edge_detection as qed



# Insert API key generated after registring in IBM Quantum Experience
# IBMQ.save_account('API KEY')

IBMQ.load_account()
provider = IBMQ.get_provider( group='open', project='main')


# dimensions of the image
size=32

#target image
image=utils.get_Cat_image()

#normalized image
normalized_image=utils.image_normalization(image,32,True)

#get target image pixel values for comparison with output image
img_arr=utils.get_image_pixel_value(image,32)


# initialize qubits and classical registers for building the circuit
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)

# create circuit
qc = QuantumCircuit(anc, img, anc2, c)

# apply hadamard gates
for i in range(1, len(img)):
    qc.h(img[i])

# frqi circuit from https://github.com/Shedka/citiesatnight
for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[j] for j in range(1,len(img))])

#rotate the image 180 deg
# qed.quantum_rotate_image(qc)

#Edge Detection 
# qed.quantum_edge_detection()

qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000


#To add noise on the simulation UNCOMMENT BELOW LINES

# backend = provider.get_backend('ibmq_16_melbourne')
# noise_model = NoiseModel.from_backend(backend)
# # Get coupling map from backend
# coupling_map = backend.configuration().coupling_map
# # Get basis gates from noise model
# basis_gates = noise_model.basis_gates

# To run without noise UNCOMMENT BELOW LINES
# result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots,coupling_map=coupling_map,
#                  basis_gates=basis_gates,
#                  noise_model=noise_model).result()

# To run without noise UNCOMMENT BELOW LINES
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()

# Image retrieval from quantum state to pixels

genimg = np.array([])

#### decode
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])



# inverse nomalization
genimg *= size * 255.0
# genimg = np.sin(genimg)

same,notSame= utils.get_count_of_pixel(img_arr,genimg)
print(same,notSame)
percentage= (same/1024)*100
print ("Total image recovered "+ str(percentage))

# convert type
genimg = genimg.astype('int')
genimg = genimg.reshape((size,size))
plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
plt.savefig('Result'+'.png')
plt.show()