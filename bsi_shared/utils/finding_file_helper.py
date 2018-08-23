import os
import sys

def get_most_recent_file_from_dir(folder):
    return sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)[0]



def choose_file_with_stdin(folder, prompt=None):
    files = sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    if not prompt:
        prompt = 'Enter index for desired file.  Enter "r" for most recently updated'
    print(prompt)
    for f in files:
        if f.startswith('.'):
            files.remove(f)
    for i, f in enumerate(files):
        print('\t[{}]: {}'.format(i, f))
    print("Selection: "),
    # choice = sys.stdin.readline()
    choice = raw_input()
    if str(choice).strip() == 'r':
        return get_most_recent_file_from_dir(folder)
    if not choice:
        return None
    try:
        choice = int(choice)
        if choice < len(files):
            return files[choice]
    except ValueError as e:
        print('Bad Input')
        return None