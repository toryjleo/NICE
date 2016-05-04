# NICE (Northeastern Interactive Clustering Engine)
The Northeastern Interactive Clustering Engine (NICE) is an open source 
data analysis framework, which aims to helping researchers in different 
domains gain insight in their data by providing an interactive 
webpage-based interface with a set of clustering algorithms and the 
capability to visualize the clustering results. The framework is still 
under development.

## For Collaborators:
1. Git clone the repository: `git clone git@github.com:yiskylee/NICE.git`
2. Create your own local feature branch: `git checkout -b your-own-feature-branch develop`
3. Make your own feature branch visible by pushing it to the remote repo (DO NOT PUSH IT TO THE DEVELOP BRANCH): `git push origin your-own-feature-branch`
4. Develop your own feature branch in your local repository: `git add`, `git commit`, etc..
5. After your own branch is completed, make sure to merge the latest development branch to your own feature branch: 1) `git checkout your-own-feature-branch` 2) `git pull origin develop`
6. Update your own feature branch on the remote repository by: `git push origin your-own-feature-branch`
7. Make a pull request with base being develop and compare being your-own-feature-branch
8. After the pull request is merged, your-own-feature-branch on the remote repository will be soon deleted, delete it on your local repository by: `git branch -d your-own-feature-branch`

## Coding Style:
We are following Google [c++ style guide](https://google.github.io/styleguide/cppguide.html), make sure to use `google_styleguide/cpplint/cpplint.py` to check your program file before push. You can also import `google_styleguide/eclipse-cpp-google-style.xml` into Eclipse to auto-format your code.
