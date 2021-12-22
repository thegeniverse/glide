import setuptools

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    requirements_list = req_file.read().strip().split("\n")

setuptools.setup(
    name="glide",
    packages=["glide"],
    install_requires=requirements_list,
)