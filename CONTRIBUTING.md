# Contributing to NEUTRINO

We would love for you to contribute to our project and help make it even better than it is today! As a
contributor, here are the guidelines we would like you to follow:
 - [Question or Problem?](#question)
 - [Issues and Bugs](#issue)
 - [Feature Requests](#feature)
 - [Submission Guidelines](#submit)
 
## <a name="question"></a> Got a Question or Problem?

Please do not hesitate to contact the [main author by email][main-author-email].

## <a name="issue"></a> Found a Bug?

If you find a bug in the source code, you can help us by [submitting an issue](#submit-issue) to our
[GitHub Repository][github]. Even better, you can [submit a Pull Request](#submit-pr) with a fix.

## <a name="feature"></a> Missing a Feature?

You can *request* a new feature by [submitting an issue](#submit-issue) to our GitHub Repository. If you would like to
*implement* a new feature, please submit an issue with a proposal for your work first, to be sure that we can use it.

Please consider what kind of change it is:
* For a **Major Feature**, first open an issue and outline your proposal so that it can be
discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
and help you to craft the change so that it is successfully accepted into the project.
* **Small Features** can be crafted and directly [submitted as a Pull Request](#submit-pr).

## <a name="submit"></a> Submission Guidelines

### <a name="submit-issue"></a> Submitting an Issue

Before you submit an issue, please search the issue tracker, maybe an issue for your problem already exists and the
discussion might inform you of workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we need to reproduce and confirm it. In order
to reproduce bugs we will need as much information as possible, and preferably be in touch with you to gather
information.

### <a name="submit-pr"></a> Submitting a Pull Request (PR)

When you wish to contribute to the code base, please consider the following guidelines:
* Make a [fork](https://guides.github.com/activities/forking/) of this repository.
* Make your changes in your fork, in a new git branch:
     ```shell
     git checkout -b my-fix-branch master
     ```
* Create your patch, **including appropriate test cases**.
* Run the full test suite, and ensure that all tests pass.
* Commit your changes using a descriptive commit message.
     ```shell
     git commit -a
     ```
  Note: the optional commit `-a` command line option will automatically “add” and “rm” edited files.
* Push your branch to GitHub:
    ```shell
    git push origin my-fix-branch
    ```
  
[main-author-email]: mailto:michael.reimann@epfl.ch
[github]: https://github.com/MWolfR/topological_sampling