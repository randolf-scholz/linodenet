import sys

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'  # when saving svg, keep text as text
plt.rcParams['text.usetex'] = True

plt.text(.5, .5, r"\fbox{\begin{tabular}{ll}test & and \\ more &test\end{tabular}}", bbox=dict(facecolor='none'))

plt.savefig("figure.svg")
plt.savefig("figure.pdf")

print(sys.version)
print(matplotlib.__version__)
print(matplotlib.get_backend())
