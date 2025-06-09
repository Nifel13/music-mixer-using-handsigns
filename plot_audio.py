from pyo import *
import time
import threading

def get_spectrum_data(pyo_audio_object, duration=1.0, fft_size=1024, low_freq=20, high_freq=20000, 
                     update_rate=0.05, wintype=2):
    """
    Get spectrum data from a PyoObject (audio source) that's already playing.
    
    Args:
        pyo_audio_object: PyoObject - The audio source (e.g., SfPlayer, Sine, etc.)
        duration: float - How long to collect spectrum data (seconds)
        fft_size: int - FFT size, must be power of 2 (larger = better freq resolution)
        low_freq: float - Lowest frequency to analyze (Hz)
        high_freq: float - Highest frequency to analyze (Hz) 
        update_rate: float - How often to update spectrum (seconds)
        wintype: int - Window type (0=rectangular, 1=hamming, 2=hanning, etc.)
    
    Returns:
        dict: {
            'spectrum_frames': list of spectrum frames (each frame is list of lists for each channel),
            'timestamps': list of timestamps for each frame,
            'frequency_bins': list of frequency values for each bin,
            'sample_rate': sample rate used,
            'channels': number of audio channels
        }
    """
    
    # Storage for collected data
    collected_data = {
        'spectrum_frames': [],
        'timestamps': [],
        'start_time': time.time()
    }
    
    # Create spectrum analyzer from the existing audio object
    spectrum = Spectrum(
        input=pyo_audio_object,
        size=fft_size,
        wintype=wintype
    )
    
    # Configure frequency range
    spectrum.setLowFreq(low_freq)
    spectrum.setHighFreq(high_freq)
    spectrum.polltime(update_rate)
    
    def spectrum_callback(data):
        """Callback to collect spectrum data"""
        current_time = time.time() - collected_data['start_time']
        collected_data['spectrum_frames'].append(data.copy())  # Copy to avoid reference issues
        collected_data['timestamps'].append(current_time)
    
    # Set callback and start analysis
    spectrum.setFunction(spectrum_callback)
    spectrum.poll(True)
    
    # Collect data for specified duration
    time.sleep(duration)
    
    # Stop analysis
    spectrum.poll(False)
    
    # Calculate frequency bins
    server = pyo_audio_object.getServer()
    sample_rate = server.getSamplingRate()
    nyquist = sample_rate / 2
    
    # Frequency bins based on the actual bounds set by Spectrum
    actual_low = spectrum.getLowfreq()
    actual_high = spectrum.getHighfreq()
    num_bins = len(collected_data['spectrum_frames'][0][0]) if collected_data['spectrum_frames'] else fft_size // 2
    
    frequency_bins = []
    for i in range(num_bins):
        freq = actual_low + (actual_high - actual_low) * i / (num_bins - 1)
        frequency_bins.append(freq)
    
    # Determine number of channels
    num_channels = len(collected_data['spectrum_frames'][0]) if collected_data['spectrum_frames'] else 1
    
    return {
        'spectrum_frames': collected_data['spectrum_frames'],
        'timestamps': collected_data['timestamps'],
        'frequency_bins': frequency_bins,
        'sample_rate': sample_rate,
        'channels': num_channels,
        'duration': duration,
        'fft_size': fft_size,
        'low_freq': actual_low,
        'high_freq': actual_high
    }


def get_realtime_spectrum(pyo_audio_object, callback_func, fft_size=1024, low_freq=20, 
                         high_freq=20000, update_rate=0.05, wintype=2):
    """
    Get real-time spectrum data from a PyoObject with a custom callback.
    
    Args:
        pyo_audio_object: PyoObject - The audio source
        callback_func: function - Your function to handle spectrum data
                      Should accept (spectrum_data, frequency_bins, timestamp)
        fft_size: int - FFT size
        low_freq: float - Lowest frequency (Hz)
        high_freq: float - Highest frequency (Hz)
        update_rate: float - Update rate (seconds)  
        wintype: int - Window type
    
    Returns:
        Spectrum object - Call .poll(False) to stop analysis
    """
    
    spectrum = Spectrum(
        input=pyo_audio_object,
        size=fft_size,
        wintype=wintype
    )
    
    spectrum.setLowFreq(low_freq)
    spectrum.setHighFreq(high_freq)
    spectrum.polltime(update_rate)
    
    # Calculate frequency bins
    server = pyo_audio_object.getServer()
    sample_rate = server.getSamplingRate()
    
    start_time = time.time()
    
    def internal_callback(data):
        current_time = time.time() - start_time
        
        # Calculate frequency bins dynamically
        actual_low = spectrum.getLowfreq()
        actual_high = spectrum.getHighfreq()
        num_bins = len(data[0]) if data else fft_size // 2
        
        frequency_bins = []
        for i in range(num_bins):
            freq = actual_low + (actual_high - actual_low) * i / (num_bins - 1)
            frequency_bins.append(freq)
        
        # Call user's callback
        callback_func(data, frequency_bins, current_time)
    
    spectrum.setFunction(internal_callback)
    spectrum.poll(True)
    
    return spectrum


# Example usage functions
def example_usage():
    """Example of how to use the spectrum analysis functions"""
    
    # Initialize server
    s = Server().boot()
    s.start()
    
    # Create an audio source (example with a sine wave)
    # In practice, this could be SfPlayer, Input, or any PyoObject
    audio_source = SfPlayer("audios/queen.mp3", speed=1.0, loop=True)
    fft = PVAnal(audio_source, size=1024)
    pitch = PVTranspose(fft, transpo=1)
    synth = PVSynth(pitch)
    amp = synth * 0.5
    amp.out()
    
    # Method 1: Collect spectrum data for a specific duration
    print("Collecting spectrum data for 2 seconds...")
    spectrum_data = get_spectrum_data(audio_source, duration=2.0)
    
    print(f"Collected {len(spectrum_data['spectrum_frames'])} frames")
    print(f"Frequency range: {spectrum_data['low_freq']:.1f} - {spectrum_data['high_freq']:.1f} Hz")
    print(f"Number of channels: {spectrum_data['channels']}")
    
    # Method 2: Real-time processing with callback
    def my_spectrum_callback(spectrum_data, frequency_bins, timestamp):
        # Process spectrum data in real-time
        if spectrum_data:
            max_magnitude = max(spectrum_data[0])  # First channel
            max_freq_idx = spectrum_data[0].index(max_magnitude)
            dominant_freq = frequency_bins[max_freq_idx]
            print(f"Time: {timestamp:.2f}s, Dominant freq: {dominant_freq:.1f} Hz")
    
    print("\nStarting real-time analysis for 3 seconds...")
    spectrum_analyzer = get_realtime_spectrum(audio_source, my_spectrum_callback)
    
    time.sleep(3)
    spectrum_analyzer.poll(False)  # Stop analysis
    
    # Cleanup
    audio_source.stop()
    s.stop()
    s.shutdown()

if __name__ == "__main__":
    example_usage()