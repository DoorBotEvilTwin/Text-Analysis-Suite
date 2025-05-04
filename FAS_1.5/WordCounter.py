import string
from collections import Counter
import os

def count_word_frequency(filename):
    """
    Counts the frequency of each word in a text file, ignoring punctuation and case,
    prints the word frequencies to the console, and saves the sorted word list
    with counts to a new file.

    Args:
        filename (str): The path to the text file.

    Returns:
        None: Prints the word frequencies to the console and saves the sorted
              word list with counts to a file named '{source_filename}_list.txt'.
              Handles file not found errors and prints an appropriate message.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:  # Explicitly specify UTF-8 encoding
            text = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()

    # Count word frequencies
    word_counts = Counter(words)

    # Print word frequencies, handling empty file case
    if not word_counts:
        print("The file is empty.")
    else:
        print("Word, # of occurrences")
        for word, count in word_counts.most_common():
            print(f"{word}, {count}")

        # Save the sorted word list with counts to a new file
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}_list.txt"
        try:
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                outfile.write("Word, # of occurrences\n")  # Add header to the output file
                for word, count in word_counts.most_common():
                    outfile.write(f"{word}, {count}\n")
            print(f"\nSorted word list with counts saved to: {output_filename}")
        except Exception as e:
            print(f"An error occurred while saving the word list: {e}")

if __name__ == "__main__":
    filename = input("Enter the path to the text file: ")
    count_word_frequency(filename)