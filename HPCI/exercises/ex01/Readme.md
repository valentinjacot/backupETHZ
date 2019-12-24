# Exercise 1

## Goals

* Learn the basics of Git
* Get familiar with accessing remote machines (`ssh`, `sshfs`, `scp`, `rsync`)
* Learn to obtain hardware information from vendors
* You can compute nominal bandwidth and peak performance of a given hardware
* Use these values to characterize hardware with the Roofline model
* You can apply the Roofline model to your software and relate it to
  performance on a given hardware
* You have an idea how the Roofline differs for CPUs and GPUs

## Grading

You can grade the exercise by using the `grade.py` grading script (Python).
You determine your point score by grading the exercise yourself, detailed
point distributions are provided with the exercise solutions.  You then create
your grade file by calling the `grade.py` script (example where with 2
questions to be graded):

```bash
python grade.py \
    --question1 12 \
    --comment1 'I justify my point score because of this and that' \
    -q2 20 \
    -c2 'I made an assumption in part b' \
    -c2 'Part c did not converge'
```

Explanation:

* Pass your point score for the question `X` using the `--questionX` argument.
  Alternatively, you can use the short form `-qX`.  The arguments accepts an
  _integer_ value.  The default value if omitted is `0`.
* **Optional**: You can decorate question `X` with comments by using the 
  `--commentX` or `-cX` argument.  You may use this argument multiple times to
  pass more than one comment if you like.  The argument accepts a _string_, be
  sure to quote them correctly.
* Use `python grade.py --help` for help.

For your convenience, we provide a `make_grade.sh` BASH script where you can
fill in your point scores and save the file for your later reference.  The
exercise grades are determined with a tight linear scale.  Zero points is a
grade 3.0, maximum points is a 6.0.

### Hand-in your grade

The script generates a `grade.txt` file for the exercise.  You hand-in your
grade by **submitting** this file on Moodle under the 'Grading' section.  You
can hand-in your grade any time before January 1, 2019.

Moodle:
[GRADE SUBMISSION](https://moodle-app2.let.ethz.ch/course/view.php?id=5072)
