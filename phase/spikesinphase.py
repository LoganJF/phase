from mat73 import loadmat as loadmat_73
from scipy.io import loadmat
import datetime
from collections.abc import Sequence
import numpy as np
import pandas as pd
import math

__all__ = ['calculate_number_spikes_in_phase_space']


def calculate_number_spikes_in_phase_space(filepath=None, save_path=None, neuron_phase_relative_to='IC',
                                           neuron_followers='all',
                                           phase_bins=100, within_burst_interval=.25, minimum_burst_duration=1,
                                           combine_close_long_bursts=True, combine_close_bursts_duration=3):
    """Calculates number of spike in phase space relative to one neuron.

    Parameters
    ----------
    filepath: str, must end in .mat; absolute location of the file e.g. ('/Users/loganfickling/Desktop/Test.mat')
    save_path: str, must end in .csv; absolute location to save the output(eg '/Users/loganfickling/Desktop/Test.csv')
    neuron_phase_relative_to: str, by default 'IC', name of neuron to use for making phase space relative to.
    neuron_followers: str or list, by default 'all', the list of neurons to analyze, if 'all', all neurons will be done.
    phase_bins: int, by default 100,number of bins to use when creating phase space
    within_burst_interval: float, by default .25, the maximum allowable tolerance (in seconds) between two spikes to be
        considered within the same burst. This is only used for the neuron used to create phase space (i.e. by
        default IC)
    minimum_burst_duration: float/int, by default 1, the minimum length of a burst (in seconds) to be considered valid
        activity to use in the creation of phase_space.
    combine_close_long_bursts: bool, by default True, whether or not the code should combine "close" bursts longer than
        the minimum_burst_duration. Close is determined by combine_close_bursts_duration parameter.
    combine_close_bursts_duration: float/int, by default 3, combines all [long] burst onsets if their onsets are
        separated by less than this value (unit is time in seconds).

    Returns
    ----------
    Returns a pandas DataFrame if save_path=None, otherwise returns nothing but saves the output in a .csv file at
        the absolute location of save_path. Two files are saved, one presenting all bins across the various "trials" of
        the activity (save_path + _individual_trials.csv) and the other representing only the summed value across all
        the "trials" at each bin (save_path + _summed.csv).

    Important Notes
    ---------
    Function will not work properly if multiple separate conditions are ran at the same time. For each experiment, pls
    separately run each.
    """
    struct = None

    agg_df, df = _calculate_number_spikes_in_phase_space(filepath=filepath, struct=struct,
                                                         neuron_phase_relative_to=neuron_phase_relative_to,
                                                         neuron_followers=neuron_followers,
                                                         phase_bins=phase_bins,
                                                         within_burst_interval=within_burst_interval,
                                                         minimum_burst_duration=minimum_burst_duration,
                                                         combine_close_long_bursts=combine_close_long_bursts,
                                                         combine_close_bursts_duration=combine_close_bursts_duration)
    if agg_df is None:  # handling if something crashed the script
        return

    if save_path is None:
        save_path = filepath.replace('.mat', '.csv')

    if save_path is not None:
        if '.csv' not in save_path:
            save_path = save_path + '.csv'
        df.to_csv(save_path.replace('.csv', '_individual_trials.csv'))
        print('Successfully saved file at {}'.format(save_path.replace('.csv', '_individual_trials.csv')))
        agg_df.to_csv(save_path.replace('.csv', '_summed.csv'))
        print('Successfully saved file at {}'.format(save_path.replace('.csv', '_summed.csv')))

    # else:
    # return agg_df, df


def find_nearest(array, value, return_index_not_value=True, is_sorted=True):
    """Given an array and a value, returns either the index or value of the nearest match

    Parameters
    ----------
    array: np.array, array of values to check for matches
    value: int/float, value to find the closest match to
    return_index_not_value: bool, whether to return the index(True) or the value (False)
        of the found match
    is_sorted: bool, whether the array is sorted in order of values

    Returns
    -------
    Either the index or value of the nearest match
    """
    if is_sorted:
        idx = np.searchsorted(array, value, side='right')

        if ((idx > 0) and (idx == len(array)) or (math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx]))):
            if not return_index_not_value:
                return array[idx - 1]

            if return_index_not_value:
                return idx - 1

        else:
            if not return_index_not_value:
                return array[idx]

            if return_index_not_value:
                return idx

    elif not is_sorted:
        idx = (np.abs(array - value)).argmin()

        if not return_index_not_value:
            return array[idx]

        if return_index_not_value:
            return idx


def _load_data_from_matlab(filepath=None, struct=None):
    """Given the filepath, loads matlab file into python

    Parameters
    ----------
    filepath: str, must end in .mat; absolute location of the file e.g. ('/Users/loganfickling/Desktop/Test.mat')
    struct: mlab struct, represents the structure loaded from path

    Returns
    -------
    struct: dictionary of dictionaries represent mlab object
    """
    if struct is None:  # If no matlab structure was pre-loaded

        if filepath is None:  # A filepath must be given without pre-loading
            raise TypeError('Please provided either the filepath or the loaded matlab structure')

        try:  # Loading data from the filepath using scipy.io.loadmat for matlab <v.7.3
            struct = loadmat(filepath)

        except NotImplementedError:  # Likely matlab v.7.3 HDF issues
            try:
                struct = loadmat_73(filepath)

            except Exception as e:
                print('Sorry failed to load data using the filepath.\nPerhaps preload it as a structure instead?')
                raise e

    return struct


def _file_header_parser_srf(string, return_all=True):
    """Parses relevant experimental data from the mlab struct's keys.

    Parameters
    ----------
    string: str, the string to be parsed corresponding to struct object key headers
    return_all: bool, if true return date, condition, and neuron else just return neuron

    Returns
    -------
    3 strings representing date, condition, and neuron values in header or just the neuron value header
    """
    date, condition, neuron = string.split('_')
    neuron = neuron.replace('spk', '').replace('lvl', '')
    try:
        if return_all:
            exper = date[3:]
            date = datetime.datetime.strptime(date[3:], '%m%d%y')
            date = date.strftime('%m-%d-%Y')
            return date, condition, neuron
    except:
        print('Failed to parse date from header, please remember to enter values manually in the .csv file')
        date = 'Please add'
        return date, condition, neuron

    return neuron


def _combine_close_bursts(burst_onset, burst_offset, combine_close_bursts_duration=3, minimum_burst_duration=1):
    """Combines long bursts of a sufficient duration together for analysis if time between them is sufficiently small

    Parameters
    ----------

    burst_onset: np.array, represents time in seconds of when the long bursts starts (of the neuron used to create
        phase space)
    burst_offset: np.array, represents time in seconds of when the long bursts stops (of the neuron used to create
        phase space)
    combine_close_bursts_duration: bool, by default True, whether or not the code should combine "close" bursts longer
        than the minimum_burst_duration. Close is determined by combine_close_bursts_duration parameter.
    minimum_burst_duration: float/int, by default 1, the minimum length of a burst (in seconds) to be considered valid
        activity to use in the creation of phase_space.

    Returns
    -------
    new_onsets: np.array, represents time in seconds of when the long bursts starts (of the neuron used to create
        phase space)
    new_offsets: np.array, represents time in seconds of when the long bursts stops (of the neuron used to create
        phase space)
    """
    burst_dur = burst_offset - burst_onset
    long_burst_locs = burst_dur > minimum_burst_duration
    locs_valid_combine_with = np.where(np.diff(burst_onset[long_burst_locs]) > combine_close_bursts_duration)
    locs_valid_combine_with = locs_valid_combine_with[0] + 1
    try:
        new_offsets = np.split(burst_offset[long_burst_locs], locs_valid_combine_with)
        new_offsets = np.array([x[-1] for x in new_offsets])
        new_onsets = np.split(burst_onset[long_burst_locs], locs_valid_combine_with)
        new_onsets = np.array([x[0] for x in new_onsets])
        return new_onsets, new_offsets
    # Handling in case there aren't failed bursts...
    except IndexError as e:
        return None, None


def _determine_burst_structure(struct, match, within_burst_interval=.25, minimum_burst_duration=1,
                               combine_close_long_bursts=True, combine_close_bursts_duration=3):
    """Calculates burst structure based upon spike data

    Parameters
    ----------
    struct: mlab struct, represents the structure loaded from path
    match: str, represents the key which matches for the struct object
    within_burst_interval: float, by default .25, the maximum allowable tolerance (in seconds) between two spikes to be
        considered within the same burst. This is only used for the neuron used to create phase space (i.e. by
        default IC)
    minimum_burst_duration: float/int, by default 1, the minimum length of a burst (in seconds) to be considered valid
        activity to use in the creation of phase_space.
    combine_close_long_bursts: bool, by default True, whether or not the code should combine "close" bursts longer than
        the minimum_burst_duration. Close is determined by combine_close_bursts_duration parameter.
    combine_close_bursts_duration: float/int, by default 3, combines all [long] burst onsets if their onsets are
        separated by less than this value.

    Returns
    -------
    burst_onset: np.array, represents time in seconds of when the long bursts starts (of the neuron used to create
        phase space)
    burst_offset: np.array, represents time in seconds of when the long bursts stops (of the neuron used to create
        phase space)

    """
    # Load Data, determine ISI, create burst structure from spikes
    neuron_phase_rel_to_spikes_d = struct[[key for key in struct.keys() if match in key.lower()][0]]
    spike_times = np.array(neuron_phase_rel_to_spikes_d['times'])
    inter_spike_interval = np.append(np.diff(spike_times), np.nan)
    locs_define_burst_structure = np.where(inter_spike_interval > within_burst_interval)
    locs_define_burst_structure = locs_define_burst_structure[0] + 1  # Formatting for split function
    bursts = np.split(spike_times, locs_define_burst_structure)
    _burst_onset = np.array([_spike_times[0] for _spike_times in bursts])
    _burst_offset = np.array([_spike_times[-1] for _spike_times in bursts])
    # Combine close bursts together into a long bursts
    if combine_close_long_bursts:

        burst_onset, burst_offset = _combine_close_bursts(burst_onset=_burst_onset,
                                                          burst_offset=_burst_offset,
                                                          combine_close_bursts_duration=combine_close_bursts_duration,
                                                          minimum_burst_duration=minimum_burst_duration)

        if burst_onset is None:
            neuron = match.replace('spk', '').replace('lvl', '').upper()

            _index_error_handling(within_burst_interval=within_burst_interval,
                                  neuron=neuron)

        return burst_onset, burst_offset

    # If the bursts aren't combined
    return _burst_onset, _burst_offset


def _calculate_phase_space_from_bursts(burst_onset, burst_offset, phase_bins=100):
    """Creates an array of time values that represents the bins between each relevant onset and offset

    Parameters
    ----------
    burst_onset: np.array, represents time in seconds of when the long bursts starts (of the neuron used to create
        phase space)
    burst_offset: np.array, represents time in seconds of when the long bursts stops (of the neuron used to create
        phase space)
    phase_bins: int, by default 100,number of bins to use when creating phase space

    Returns
    ----------
    phase_arr: np.array, represents time in seconds the correspond to each bin for each of the relevant burst onsets

    """
    phase_dictionary = {}

    for i, start_value in enumerate(burst_onset):
        try:
            onset_next_burst = burst_onset[i + 1]
        except IndexError:
            continue

        step = (onset_next_burst - start_value) / phase_bins
        arr = np.arange(start_value, onset_next_burst, step=step)

        if len(arr) != phase_bins:
            arr = arr[:phase_bins]
            assert len(arr) == phase_bins

        phase_dictionary[i] = arr

    phase_arr = np.concatenate(list(phase_dictionary.values()), dtype='object')
    return phase_arr


def _calc_total_spikes_from_phase_arr(neuron_list, all_spikes, phase_arr, phase_bins, date, condition):
    """Calculates total number of spikes given spike times and phase times

    Parameters
    ----------

    neuron_list: list, list of neurons representing the passed spike data
    all_spikes: nested nd.array, represents time (in seconds) of spikes for each neuron in neuron list
    phase_arr: nested nd.array, represents phase across multiple trials for the phase creator neuron.
    phase_bins: int, by default 100,number of bins to use when creating phase space
    date: str, the date of the experiment
    condition: str, the condition of the experiment

    Returns
    ----------
    df, pd.DataFrame representing spikes per phase per neuron
    """
    exp_data = []
    agg_exp_data = []

    for i, (neuron, data) in enumerate(zip(neuron_list, all_spikes)):
        # Summing across all trials
        nearest_phase_space = list(map(lambda x: find_nearest(array=phase_arr, value=x), data))
        df = pd.DataFrame(nearest_phase_space, columns=['bins'])
        df['bins'] %= phase_bins
        df['# of Spikes'] = 1
        df = df.groupby('bins', sort=False, as_index=False).agg({'# of Spikes': np.sum})
        df = df.reset_index(drop=True)
        #df = df.set_index(df['bins']).reindex(np.arange(100)).fillna(0)  # Formatting to add in 0 values
        df['neuron'] = neuron
        df['date'] = date
        df['condition'] = condition
        agg_exp_data.append(df)

        # Leaving each trial separate
        df = pd.DataFrame(nearest_phase_space, columns=['bins'])
        df['# of Spikes'] = 1  # Set to 1 so we can sum
        df['neuron'] = neuron
        df['date'] = date
        df['condition'] = condition
        df = df.groupby(['bins', 'neuron', 'date', 'condition'], sort=False, as_index=False).agg({'# of Spikes': np.sum})
        df['trial number'] = df['bins'] // phase_bins
        df['bins'] %= phase_bins
      
        exp_data.append(df)

    agg_df = pd.concat(agg_exp_data).sort_values(['neuron', 'bins'])
    agg_df = agg_df.reset_index(drop=True)
    df = pd.concat(exp_data).sort_values(['neuron', 'bins'])
    df = df.reset_index(drop=True)

    return agg_df, df


def _calculate_number_spikes_in_phase_space(filepath=None, struct=None, neuron_phase_relative_to='IC',
                                            neuron_followers='all',
                                            phase_bins=100, within_burst_interval=.25, minimum_burst_duration=1,
                                            combine_close_long_bursts=True, combine_close_bursts_duration=3):
    """Underlying function that actually calculates the number of spike in phase space relative to one neuron.

    Parameters
    ----------
    filepath: str, must end in .mat; absolute location of the file e.g. ('/Users/loganfickling/Desktop/Test.mat')
    struct: mlab struct, represents the structure loaded from path
    neuron_phase_relative_to: str, by default 'IC', name of neuron to use for making phase space relative to.
    neuron_followers: str or list, by default 'all', the list of neurons to analyze, if 'all', all neurons will be done.
    phase_bins: int, by default 100,number of bins to use when creating phase space
    within_burst_interval: float, by default .25, the maximum allowable tolerance (in seconds) between two spikes to be
        considered within the same burst. This is only used for the neuron used to create phase space (i.e. by
        default IC)
    minimum_burst_duration: float/int, by default 1, the minimum length of a burst (in seconds) to be considered valid
        activity to use in the creation of phase_space.
    combine_close_long_bursts: bool, by default True, whether or not the code should combine "close" bursts longer than
        the minimum_burst_duration. Close is determined by combine_close_bursts_duration parameter.
    combine_close_bursts_duration: float/int, by default 3, combines all [long] burst onsets if their onsets are
        separated by less than this value.

    Returns
    ----------
    Returns a pandas DataFrame of spikes per bin per neuron
    """
    struct = _load_data_from_matlab(filepath=filepath, struct=struct)
    neuron_list = np.array([_file_header_parser_srf(key, False) for key in struct.keys()])
    _, unique_indices = np.unique(neuron_list, return_index=True)
    neuron_list = neuron_list[unique_indices]
    date, condition, _ = _file_header_parser_srf(list(struct.keys())[0])
    all_spikes = np.array([struct[key]['times'] for key in struct.keys() if 'spk' in key.lower()], dtype='object')

    match = '{}spk'.format(neuron_phase_relative_to).lower()
    burst_onset, burst_offset = _determine_burst_structure(struct=struct,
                                                           match=match,
                                                           within_burst_interval=within_burst_interval,
                                                           minimum_burst_duration=minimum_burst_duration,
                                                           combine_close_long_bursts=combine_close_long_bursts,
                                                           combine_close_bursts_duration=combine_close_bursts_duration)
    if burst_onset is None:  # Handling if _determine_burst_structure failed
        return None, None

    phase_arr = _calculate_phase_space_from_bursts(burst_onset=burst_onset,
                                                   burst_offset=burst_offset,
                                                   phase_bins=phase_bins)

    if neuron_followers != 'all':
        if (isinstance(neuron_followers, np.ndarray) | (isinstance(neuron_followers, Sequence))):
            neuron_list = neuron_followers

    agg_df, df = _calc_total_spikes_from_phase_arr(neuron_list=neuron_list,
                                                   all_spikes=all_spikes,
                                                   phase_arr=phase_arr,
                                                   phase_bins=phase_bins,
                                                   date=date,
                                                   condition=condition)
    # Add in 0 values where no spikes occur
    agg_df = fixer_func(df=agg_df)
    df = fixer_func(df)

    return agg_df, df


def _index_error_handling(within_burst_interval=.25, neuron=None):
    """Helper function to give more informative error when user sets PR wbi rather than gastric for gastric mill neurons

    Parameters
    ----------
    within_burst_interval: float, by default .25s, time in
    neuron: str, represents the neuron phase space is made relative to

    Returns
    ----------
    Error handling
    """
    print('Encountered an IndexError while trying to determine bursts structures:')
    wbi = within_burst_interval
    gastric_neurons = ['LG', 'DG', 'GM']
    gastro_py_neurons = ['IC', 'LPG', 'PD', 'PY', 'LP', 'MG', 'AM']
    if neuron.upper() in gastric_neurons:
        if wbi < 2:
            print('For gastric mill neurons please set within_burst_interval to be >=2, currently at {}'.format(wbi))
            print('Ending analysis')

def fixer_func(df):
    """Explicitly adds in 0 values when no spikes occurs in a bin

    :param df: pd.DataFrame, either the dataframe aggregate or individual trials
    :return: dataframe with 0s
    """
    _dataframe = []

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')

    if 'trial number' in df.columns:
        for i, _df in df.groupby(['neuron', 'trial number']):
            neuron, date, cond, trial_num = _df.iloc[0][['neuron', 'date', 'condition', 'trial number']]
            _df = _df.set_index(_df['bins']).reindex(np.arange(100)).fillna(0)
            _df['neuron'] = neuron
            _df['date'] = date
            _df['condition'] = cond
            _df['trial number'] = trial_num
            #_df = _df.drop(columns='Unnamed: 0')
            _df = _df.reset_index(drop=True)
            _df['bins']=np.arange(100)
            _dataframe.append(_df)
    else:
        for i, _df in df.groupby(['neuron']):
            neuron, date, cond = _df.iloc[0][['neuron', 'date', 'condition']]
            _df = _df.set_index(_df['bins']).reindex(np.arange(100)).fillna(0)
            _df['neuron'] = neuron
            _df['date'] = date
            _df['condition'] = cond
            _df = _df.reset_index(drop=True)
            #_df = _df.drop(columns='Unnamed: 0')
            _df['bins']=np.arange(100)
            _dataframe.append(_df)

    return pd.concat(_dataframe)
