import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lenscorp',
    version='0.0.1',
    author='LENS Corporation',
    author_email='solutions@lenscorp.ai',
    description='AI Toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=['requests', 'tqdm', 'numpy',
                      'matplotlib', 'seaborn', 'scipy'],
)
