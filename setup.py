from setuptools import find_packages, setup
from typing import List
HYPHEN_E_DOT = '-e .'# The string that is used to specify the current directory in the requirments file
def get_requirments(file_path:str)->list[str]:
    '''Gets the list  requirements from the file'''
    requirements = []
    with open(file_path, 'r') as files_obj:
        requirements = files_obj.readlines()
        requirements = [req.replace('\n','') for req  in requirements]#A comprehensive list containing requirments seperated by blanks('') and not new lines('\n')

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)#removes the -e . from the list of requirments

    return requirements
setup(
    name = 'mlproject',
    version = '0.0.1',
    author='phill',
    author_email='philemonsang1532@gmail.com',
    packages =find_packages(),
    install_requires=get_requirments('requirements.txt')
)