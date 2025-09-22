
# Make sure you have enough disk place (e.g. 5~10 GB)

1. Unzip the audioset-processing-master.zip on other folder then run it
   I revised some code on process.py & utils.py  
   Original file is on Github: xrick/uec-ai-dev/datasets/codes/audioset_utilities/google_audioset_download/audio_processing

2. Open the terminal, run  pip install yt-dlp

3. Download FFMPEG (windows or Linux) and add the path to the System Variables in windows "Edit System Environment Variables"
   Then I suggest you run on vitual environment
    # Create virtual environment with Python 3.10+
      py -3.10 -m venv audioset_env

    # Activate it
      audioset_env\Scripts\activate

    # Install required packages
      pip install yt-dlp subprocess32

4. To avoid "403 Forbidden", suggest your download CloudFlare https://one.one.one.one/ for private connection.

5. Same reason as above, to avoid "403 Forbidden"
   Navigate to your project folder: for example   cd ..\comp385\assignment\GoodleAudioSet_testing
   Open cmd or powershell
   Run the following command to generate the cookies file using your default browser:
         yt-dlp --cookies-from-browser firefox --cookies cookies.txt
   
   Note: Replace firefox with your browser (e.g., chrome, edge, brave, chromium).
         This will create a cookies.txt file containing your login information, 
         which will allow the downloader to access the restricted videos.
   
6. Back to the compiler, on terminal, run  
        python process.py download -c "Alarm clock" "Siren" "Fire engine, fire truck (siren)" "Car passing by" "Vehicle horn, car horn, honking" "Motor vehicle (road)" -d E:/GoogleAudioSet/raw_audio
   Edit E:/... as your own GoogleAudioSet path. 
   Change "Siren"... as what type of raw audio files you want. (GoogleAudioSet contain 527 different labels)
   
7. After 0.5 ~ 1 hour, the terminal show you look like robot and couldn't download files',
   Step 1: Install a browser extension to export cookies   
    For Chrome: Install "Get cookies.txt LOCALLY" extension
    For Firefox: Install "cookies.txt" extension

   Step 2: Export YouTube cookies
    Go to YouTube.com and make sure you're logged in
    Use the extension to export cookies to a file called cookies.txt
    Save the cookies.txt file in the same directory as your process.py
    
8. Open new window for "GoodleAudioSet_testing" to extract feature.
   Run train_model.py for training the model
   And then test on predict_sounds.py
   
