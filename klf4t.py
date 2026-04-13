import sys
sys.path.insert(0, "/gpfs/home/cam/cam394520/ejcproj/OpenCGChromatin/src")

import numpy as np
import openmm as mm
from openmm import unit
from openmm import app

from OpenCGChromatin.biomolecules import NucleosomeArray
from OpenCGChromatin.system_building import get_system
from OpenCGChromatin.biomolecules import MDP   

import openmm.unit as unit
import numpy as np

PLATFORM = mm.Platform.getPlatformByName("CUDA")
PROPERTIES = {"CudaPrecision": "mixed"}

# Function to generate ACTG repeats of arbitrary length
def generate_actg_sequence(n):
    pattern = 'ACTG'
    return (pattern * (n // len(pattern) + 1))[:n]

# Function to generate a combination of ACTG linkers/w601 nucleosomal DNA corresponding to 
# a nucleosome array with evenly spaced nucleosome dyads
def generate_even_dyads_sequence(N_nucleosomes, linker_length):
    w601 = 'ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT'
    DNA_sequence = generate_actg_sequence(linker_length//2)
    for _ in range(N_nucleosomes-1):
        DNA_sequence += w601 + generate_actg_sequence(linker_length)
    DNA_sequence += w601 + generate_actg_sequence(linker_length//2)
    return DNA_sequence

# Set the length of linker DNA
linker_length = 30

# Get the DNA sequence for a regularly spaced array of 6 nucleosomes with 30 bp linker
DNA_sequence = generate_even_dyads_sequence(6, linker_length)

# Set the nucleosome sequence - default core with all tails attached, labelled 1kx5
nucleosome_sequence = 6 * ['1kx5']

# Build the dyad positions list
first_dyad = linker_length//2 + 73
dyad_positions = [first_dyad]
for i in range(1, 6):
    dyad_positions.append(first_dyad + i * (147 + linker_length))

# Instantiate a NucleosomeArray object, relax it and extract the Topology and relaxed coordinates
nucleosome_array = NucleosomeArray(nucleosome_sequence, DNA_sequence, dyad_positions)
nucleosome_array.relax()

na_topology = nucleosome_array.topology
na_positions = nucleosome_array.relaxed_coords

# Add KLF4
KLF4_sequence = 'MRQPPGESDMAVSDALLPSFSTFASGPAGREKTLRQAGAPNNRWREELSHMKRLPPVLPGRPYDLAAATVATDLESGGAGAACGGSNLAPLPRRETEEFNDLLDLDFILSNSLTHPPESVAATVSSSASASSSSSPSSSGPASAPSTCSFTYPIRAGNDPGVAPGGTGGGLLYGRESAPPPTAPFNLADINDVSPSGGFVAELLRPELDPVYIPPQQPQPPGGGLMGKFVLKASLSAPGSEYGSPSVISVSKGSPDGSHPVVVAPYNGGPPRTCPKIKQEAVSSCTHLGAGPPLSNGHRPAAHDFPLGRQLPSRTTPTLGLEEVLSSRDCHPALPLPPGFHPHPGPNYPSFLPDQMQPQVPPLHYQGQSRGFVARAGEPCVCWPHFGTHGMMLTPPSSPLELMPPGSCMPEEPKPKRGRRSWPRKRTATHTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRHF'

# Globular domain: zinc fingers, residues 4
globular = [list(range(429, 513))]
protein = MDP("KLF4", KLF4_sequence, globular, 'AF-O43474-F1-model_v6.pdb')

protein.relax()
p_positions = protein.relaxed_coords  

p_topology = protein.topology

# Initialise the Modeller with the nucleosome array
model = app.Modeller(na_topology, na_positions * unit.nanometer)


n_klf4 = 0
count = 0
for i in range(3):
    for j in range(3):
        for k in range(3):
            if count >= n_klf4:
                break
            if [i, j, k] != [0, 0, 0]:
                offset = 15 * np.array([i, j, k])
                modified_p_positions = p_positions + offset
                model.add(p_topology, modified_p_positions * unit.nanometer)
                count += 1

# Set vectors for periodic box and apply them to the model Topology
box_vecs = 100 * np.eye(3)  # cubic box with 100 nm side length
model.topology.setPeriodicBoxVectors(box_vecs)

# Save a PDB file containing the initial model - inspect in VMD before analysing!
app.PDBFile.writeFile(model.topology, model.positions, open('start_model.pdb', 'w'))

# Convert positions to numpy array without OpenMM units
multi_positions = np.array([position.value_in_unit(unit.nanometer) for position in model.positions])
multi_topology = model.topology

# Create an OpenMM System object for the combined model
system = get_system(multi_positions,
                    multi_topology,
                    {nucleosome_array.chain_id: nucleosome_array.globular_indices,
                     protein.chain_id: protein.globular_indices},
                    [nucleosome_array.dyad_positions, None])

# Set up simulation
integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 0.01/unit.picosecond, 10*unit.femtosecond)
simulation = app.Simulation(multi_topology, system, integrator, platform=PLATFORM, platformProperties=PROPERTIES)

# Set positions and box vectors, then minimise
simulation.context.setPositions(multi_positions)
simulation.context.setPeriodicBoxVectors(*box_vecs)
simulation.minimizeEnergy()

# Create reporters
xtc_reporter = app.XTCReporter('traj.xtc', 100000)
state_reporter = app.StateDataReporter('state_data.out', reportInterval=10000, step=True,
                                        potentialEnergy=True, temperature=True, elapsedTime=True)
total_step = int(4000*unit.nanosecond/(10*unit.femtosecond))
checkpoint = app.CheckpointReporter('checkpoint.chk', total_step)
simulation.reporters = [xtc_reporter, state_reporter, checkpoint]

# Run 4 µs simulation
simulation.step(total_step)