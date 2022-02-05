import os  # https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
import shutil
import SA1

# define paths for folders to store unprocessed and processed data files
WorkingDirectoryFolderPath = "C:\\Users\\Scantlebury_Lab\\PycharmProjects\\Scantleburylab\\"

UnprocessedRecordingDataFolderPath = WorkingDirectoryFolderPath + "UnprocessedData\\"
ProcessedRecordingDataFolderPath = WorkingDirectoryFolderPath + "ProcessedData\\"
FilteredDataFolderPath = ProcessedRecordingDataFolderPath + "FilteredData\\"
UnfilteredDataFolderPath = ProcessedRecordingDataFolderPath + "UnfilteredData\\"
FilterCoeffDesignedInMatlab = WorkingDirectoryFolderPath + "A_B_D_G_T_H.txt"

# define a list to hold all files found
ListOffilesToBeProcessed = []

# go into unprocessed folder and find all the txt files to be processed, store their full paths in a list
for file in os.listdir(UnprocessedRecordingDataFolderPath):
    if file.endswith(".txt"):
        ListOffilesToBeProcessed.append(os.path.join(UnprocessedRecordingDataFolderPath, file))

# if List of files to be processed is not empty, process them using SeizureAnalysis function
# otherwise state there are no files to be processed
if ListOffilesToBeProcessed:
    for PathOfFileToBeProcessed in ListOffilesToBeProcessed:
        # Analysis each file in the list of tiles to be processed
        SA1.SeizureAnalysis(PathOfFileToBeProcessed, FilteredDataFolderPath, FilterCoeffDesignedInMatlab)

        # Move all the processed files from the unprocessedData folder into the ProcessedData folder under the
        # UnfilteredData folder: https://stackoverflow.com/questions/8858008/how-to-move-a-file
        shutil.move(PathOfFileToBeProcessed, UnfilteredDataFolderPath + PathOfFileToBeProcessed.split("\\")[-1])
else:
    print("No files to be processed...")
