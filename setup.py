from setuptools import find_packages,setup
from typing import List


Hypen_E_Dot="-e ."
def get_requirements(file_path:str)->List[str]:
    """
    this fucntion will return the list of requirements
    """
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]

        if Hypen_E_Dot in requirements:
            requirements.remove(Hypen_E_Dot)
        
        return requirements 


setup(
    name="ML_Project",
    version="0.0.1",
    author="Swapnil",
    author_email="swapniltomar27@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)