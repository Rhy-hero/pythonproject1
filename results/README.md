# Local results directory

The repository contains code and documentation only. Existing execution logs,
computed arrays, and generated images are provided separately in the assignment
submission package and are not published here.

To generate your own results, first copy this repository into the teacher's
student package, keeping the provided weights/ and assets/ directories, then run:

```bash
python check.py --stage all > results/check_all.txt
python run.py --output results/output_tokens.npz
python validate_additional.py
```

Do not upload the course-provided weights or reference arrays.
