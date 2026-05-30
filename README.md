# Machine-Learning-for-Social-Media-Influence

What this script does

Think of the spreadsheet as a messy notebook full of information. The script acts like a helper that tidies everything before you use it.

Opens the spreadsheet
It reads the file sm.csv so it can work with the data inside.
Finds missing information
Words such as NA, ?, NULL, null, and unknown are treated as "information not available" instead of actual data.
Fixes text mistakes
Removes extra spaces.
Makes labels consistent. For example, if one row says Img and another says image, both are changed to the same format.
Fixes numbers
Sometimes numbers are stored as text. The script converts them into real numbers so calculations can be performed correctly.
Removes repeated entries
If the exact same row appears more than once, it keeps only one copy.
Fills in missing numbers
If a numeric value is missing, the script replaces it with a reasonable middle value from that column.
Fills in missing categories
If a category is blank, the script puts in a default value so nothing is left empty.
Saves the cleaned data
After everything is fixed, it creates a new file called sm_preprocessed.csv.
Why the old version had a problem

Previously, the script knew that values like NA and NULL meant "missing data," but it did not recognize unknown.

So if a cell contained unknown, the script treated it as normal text instead of missing information. Because of that, those cells were not cleaned or filled properly.

The updated script now recognizes unknown as missing data too, so those values are handled correctly during cleaning.
