% prepare audio and plots
%source:
%http://stackoverflow.com/questions/6681063/programming-in-matlab-how-to-process-in-real-time
its=25; % Number of time blocks in each signal
Fs = 16*1024;                    % sampling frequency in Hz
T = 0.5;                      % length of one time block in sec
t = (0:(1/Fs):(T*its))-1/Fs;      % Time vector
nfft = 2^nextpow2(Fs);        % n-point DFT
numUniq = ceil((nfft+1)/2);   %# half point
f = (0:numUniq-1)'*Fs/nfft;   %'# frequency vector (one sided)

% prepare plots
figure
hAx(1) = subplot(211);
hLine(1) = line('XData',t, 'YData',nan(size(t)), 'Color','b', 'Parent',hAx(1));
xlabel('Time (s)'), ylabel('Amplitude')
hAx(2) = subplot(212);
hLine(2) = line('XData',f, 'YData',nan(size(f)), 'Color','b', 'Parent',hAx(2));
xlabel('Frequency (Hz)'), ylabel('Magnitude (dB)')
set(hAx, 'Box','on', 'XGrid','on', 'YGrid','on')
%specgram(sig, nfft, Fs);

% prepare audio recording
recObj = audiorecorder(Fs,24,1);
sigAll=zeros(length(t),1);
sigStart=1;
sigEnd=floor(T*Fs);










