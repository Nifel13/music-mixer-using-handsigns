from pyo import *
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

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


class LiveSpectrumPlotter:
    """Real-time spectrum analyzer with live matplotlib visualization"""
    
    def __init__(self, pyo_audio_object, fft_size=1024, low_freq=20, high_freq=5000, 
                 update_rate=0.05, history_length=100):
        self.pyo_audio_object = pyo_audio_object
        self.fft_size = fft_size
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.update_rate = update_rate
        self.history_length = history_length
        
        # Data storage
        self.spectrum_history = deque(maxlen=history_length)
        self.frequency_bins = []
        self.current_spectrum = []
        self.is_running = False
        
        # Setup spectrum analyzer
        self.spectrum = None
        self.setup_spectrum_analyzer()
        
        # Setup plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.setup_plots()
        
    def setup_spectrum_analyzer(self):
        """Initialize the Pyo spectrum analyzer"""
        self.spectrum = Spectrum(
            input=self.pyo_audio_object,
            size=self.fft_size,
            wintype=2  # Hanning window
        )
        
        self.spectrum.setLowFreq(self.low_freq)
        self.spectrum.setHighFreq(self.high_freq)
        self.spectrum.polltime(self.update_rate)
        self.spectrum.setFunction(self.spectrum_callback)
        
    def spectrum_callback(self, data):
        """Callback function to receive spectrum data from Pyo"""
        if data and len(data) > 0:
            # Use first channel and ensure it's a flat list of numbers
            spectrum_data = data[0]
            
            # Convert to list of floats if needed
            if hasattr(spectrum_data, '__iter__'):
                try:
                    self.current_spectrum = [float(x[1]) for x in spectrum_data]
                except (TypeError, ValueError):
                    print(f"Warning: Unable to convert spectrum data to floats: {type(spectrum_data)}")
                    print(spectrum_data)
                    return
            else:
                print(f"Warning: Unexpected spectrum data format: {type(spectrum_data)}")
                return
            
            # Add to history for spectrogram
            self.spectrum_history.append(self.current_spectrum.copy())
            
            # Calculate frequency bins if not done yet
            if not self.frequency_bins:
                actual_low = self.spectrum.getLowfreq()
                actual_high = self.spectrum.getHighfreq()
                num_bins = len(self.current_spectrum)
                
                self.frequency_bins = []
                for i in range(num_bins):
                    freq = actual_low + (actual_high - actual_low) * i / (num_bins - 1)
                    self.frequency_bins.append(freq)
    
    def setup_plots(self):
        """Setup matplotlib plots"""
        # Top plot: Real-time spectrum
        self.ax1.set_title('Live Spectrum Analyzer', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Frequency (Hz)')
        self.ax1.set_ylabel('Magnitude')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(self.low_freq, self.high_freq)
        
        # Bottom plot: Spectrogram
        self.ax2.set_title('Spectrogram (Time vs Frequency)', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Time (frames)')
        self.ax2.set_ylabel('Frequency (Hz)')
        
        # Initialize empty plots
        self.line, = self.ax1.plot([], [], 'b-', linewidth=1.5)
        self.spectrogram_img = None
        
        plt.tight_layout()
        
    def update_plot(self, frame):
        """Animation update function"""
        if not self.current_spectrum or not self.frequency_bins:
            return self.line, 
            
        # Update spectrum plot
        self.line.set_data(self.frequency_bins, self.current_spectrum)
        
        # Auto-scale y-axis for spectrum
        if self.current_spectrum:
            # Ensure we're working with numeric values
            try:
                # Handle case where current_spectrum might be nested lists
                if isinstance(self.current_spectrum[0], (list, tuple)):
                    # Flatten if nested
                    flat_spectrum = [item for sublist in self.current_spectrum for item in sublist]
                    max_val = max(flat_spectrum)
                else:
                    max_val = max(self.current_spectrum)
                
                # Only set ylim if max_val is a number
                if isinstance(max_val, (int, float)):
                    self.ax1.set_ylim(0, max_val * 1.1)
            except (TypeError, ValueError):
                # If there's any issue with the data, use a default range
                self.ax1.set_ylim(0, 1.0)
        
        # Update spectrogram
        if len(self.spectrum_history) > 1:
            # Convert deque to numpy array for plotting
            spectrogram_data = np.array(list(self.spectrum_history)).T
            
            # Remove old image
            if self.spectrogram_img:
                self.spectrogram_img.remove()
            
            # Create new spectrogram image
            extent = [0, len(self.spectrum_history), self.low_freq, self.high_freq]
            self.spectrogram_img = self.ax2.imshow(
                spectrogram_data, 
                aspect='auto', 
                origin='lower',
                extent=extent,
                cmap='viridis',
                interpolation='nearest'
            )
        
        return self.line,
    
    def start(self):
        """Start the live spectrum analysis and plotting"""
        print("Starting live spectrum analyzer...")
        print("Close the plot window to stop.")
        
        self.is_running = True
        self.spectrum.poll(True)
        
        # Start matplotlib animation
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=int(self.update_rate * 1000),  # Convert to milliseconds
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()
        
    def stop(self):
        """Stop the spectrum analysis"""
        if self.spectrum:
            self.spectrum.poll(False)
        self.is_running = False
        print("Spectrum analysis stopped.")


# Example usage functions
def example_usage():
    """Example of how to use the spectrum analysis functions"""
    
    # Initialize server
    s = Server().boot()
    s.start()
    
    # Create an audio source (example with multiple sine waves for interesting spectrum)
    # In practice, this could be SfPlayer, Input, or any PyoObject
    audio_source = (Sine(freq=440, mul=0.2) + 
                   Sine(freq=880, mul=0.15) + 
                   Sine(freq=1320, mul=0.1) +
                   Noise(mul=0.05)).out()
    
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


def example_live_plot():
    """Example with live spectrum plotting"""
    
    # Initialize server
    s = Server().boot()
    s.start()
    
    # Option 1: Load an audio file
    # audio_source = SfPlayer("your_song.wav", loop=True, mul=0.5).out()
    
    # Option 2: Use microphone input
    # audio_source = Input(chnl=0, mul=0.5)
    
    # Option 3: Generate test audio (multiple tones + noise for interesting spectrum)
    audio_source = SfPlayer("audios/drugs.mp3", speed=1.0, loop=True)
    fft = PVAnal(audio_source, size=1024)
    pitch = PVTranspose(fft, transpo=1)
    synth = PVSynth(pitch)
    amp = synth * 0.5
    amp.out()
    
    # Create live spectrum plotter
    plotter = LiveSpectrumPlotter(
        pyo_audio_object=audio_source,
        fft_size=1024,
        low_freq=20,
        high_freq=4000,
        update_rate=0.05,  # 20 FPS
        history_length=200  # Keep 200 frames of history for spectrogram
    )
    
    try:
        # Start live plotting (this will block until plot window is closed)
        plotter.start()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        plotter.stop()
        audio_source.stop()
        s.stop()
        s.shutdown()
        print("Audio server stopped.")

if __name__ == "__main__":
    # Choose which example to run:
    
    # Run basic example
    # example_usage()
    
    # Run live plotting example
    example_live_plot()