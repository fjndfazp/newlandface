from os.path import abspath, dirname, join 
from setuptools import setup, find_packages 
​
# 获取requirements.txt里的依赖信息 
install_reqs = [req.strip() for req in open(abspath(join(dirname(__file__), 'requirements.txt')))] 
​ 
with open("README.md", 'r', encoding="utf-8") as f: 
    long_description = f.read() 
​ 
setup( 
    name='newlandface', 
    version='1.0.0', 
    description='A Lightweight Face Detection and Facial Attribute Analysis Framework (Age, Gender, Emotion) for Python', 
    author='fjndfazp', 
    author_email='gyyzp@qq.com', 
    url="https://github.com/fjndfazp/newlandface", 
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.5.5',
    install_requires=install_reqs
)