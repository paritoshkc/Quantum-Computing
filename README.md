
# Quantum-Computing
Quantum Computing has paved its path from being a theory to physical read-to-use machines. This project reflects on the implmentation of Quantum image processing with FRQI image model in Qiskit

## Getting Started



### Prerequisites

Python 3.5+ , qiskit , matplot and numpy. 
Installing Qiskit with visualization can be done using pip 
```
pip install qiskit[visualization]
```

## Running the program and seeing the result

Use runner.py to run the program and generate result.

## Selecting images to check 
There are 3 options for the image which can be selected from the Utils.py class:- 
  1. To select cat image call - util.get_Cat_image()
  2. To select MNIST Image call - util.get_MNIST_data()
  3. To select python generated image call - util.generate_image()
  
## Image transformation 
1. To rotate the image uncomment below line in runner.py
``` 
qed.quantum_rotate_image(qc)
```
2. To generate edge detection uncomment below line in runner.py
```
qed.quantum_edge_detection()
```

## Running the noise model 

To add moise model to the simulation uncomment below lines from the runner.py class
```
backend = provider.get_backend('ibmq_16_melbourne')
noise_model = NoiseModel.from_backend(backend)
coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots,coupling_map=coupling_map,
                 basis_gates=basis_gates,
                 noise_model=noise_model).result()

```

## Result
Result will be generated in the form of 'Result.png' and saved in the main folder. 

