import pickle
import matplotlib.pyplot as plt

f = open("data5_new_for_part_3.pkl","r");

rcll, prec, ap_desc_str = pickle.load(f)

plt.step(rcll, prec, color='b', alpha=0.2, where='post')
plt.fill_between(rcll, prec, step='post', alpha=0.2, color='b')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
plt.title(ap_desc_str)
plt.show()

rcll, prec, ap_desc_str = pickle.load(f)

plt.step(rcll, prec, color='b', alpha=0.2, where='post')
plt.fill_between(rcll, prec, step='post', alpha=0.2, color='b')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
plt.title(ap_desc_str)
plt.show()

rcll, prec, ap_desc_str = pickle.load(f)

plt.step(rcll, prec, color='b', alpha=0.2, where='post')
plt.fill_between(rcll, prec, step='post', alpha=0.2, color='b')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
plt.title(ap_desc_str)
plt.show()