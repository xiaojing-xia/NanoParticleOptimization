import numpy as np
from NanoParticleTools.species_data.species import Dopant

def emsInteg(x,y,startWav, endWav):
    sum_int = 0
    for _x, _y in zip(x, y):
        if ((_x>-endWav) and (_x<-startWav)):
            sum_int += _y
    return sum_int

def absInteg(x,y,startWav, endWav):
    sum_int = 0
    for _x, _y in zip(x, y):
        if ((_x<endWav) and (_x>startWav)):
            sum_int += _y
    return sum_int

def get_spectrum(doc):
    dndt = doc['data']['output']['summary'] # dndt = docs
    accumulated_dndt = {}
    for interaction in dndt:
        interaction_id = interaction[0]
        if interaction_id not in accumulated_dndt:
            accumulated_dndt[interaction_id] = []
        accumulated_dndt[interaction_id].append(interaction)
    avg_dndt = []
    for interaction_id in accumulated_dndt:

        arr = accumulated_dndt[interaction_id][-1][:-4]

        _dndt = [_arr[-4:] for _arr in accumulated_dndt[interaction_id]]

        while len(_dndt) < 1:
            _dndt.append([0 for _ in range(4)])

        mean = np.mean(_dndt, axis=0)
        std = np.std(_dndt, axis=0)
        arr.extend([mean[0], std[0], mean[1], std[1], mean[2], std[2], mean[3], std[3],])
        avg_dndt.append(arr)
    
    x = []
    y = []
    dopants = [Dopant(key, val) for key, val in doc['data']['overall_dopant_concentration'].items()]
    for interaction in [_d for _d in avg_dndt if _d[8] == 'Rad']:
        # print(interaction)
        species_id = interaction[2]
        left_state_1 = interaction[4]
        right_state_1 = interaction[6]
        ei = dopants[species_id].energy_levels[left_state_1]
        ef = dopants[species_id].energy_levels[right_state_1]

        de = ef.energy-ei.energy
        wavelength = (299792458*6.62607004e-34)/(de*1.60218e-19/8065.44)*1e9
        # print(left_state_1, right_state_1, wavelength)
        x.append(wavelength)
        y.append(interaction[10])
    return x, y

def get_int(doc, spec_range):
    x, y = get_spectrum(doc)
    return emsInteg(x, y, spec_range[0], spec_range[1])

def get_qe(doc, total_range, absorption_range):
    x, y = get_spectrum(doc)
    return emsInteg(x,y,total_range[0], total_range[1])/absInteg(x,y,absorption_range[0], absorption_range[1])