import numpy as np

N_PATIENTS = 10000
N_FEATURES = 100
RANDOM_SEED = 2023
STEP_SIZE = 1e-3

EVENT_RATE = 0.01
SIMILARITY = 0.95


def main():

	x, e1, e2 = generate_data(plot=True)

	print('Rate of event 1: %.3f' % np.mean(e1))
	print('Rate of event 2: %.3f' % np.mean(e2))
	print('Rate of co-occurrence: %.3f' % np.mean(e1 & e2))
	print('Correlation between events: %.3f' % np.corrcoef(e1, e2)[0, 1])


def generate_data(plot=False):

	rs = np.random.RandomState(RANDOM_SEED)

	# generate 100-dimensional feature vector for 10,000 patients
	x = rs.randn(N_PATIENTS, N_FEATURES)

	# generate coefficient vectors for events 1 and 2
	u1, u2 = generate_vectors_by_similarity(rs, SIMILARITY)

	# find logit offset that gives the desired event rate
	offset = find_offset(
		rs,
		np.dot(x, normed_uniform(rs, N_FEATURES)),
		EVENT_RATE,
		STEP_SIZE
	)

	# print similarity between u1 and u2
	print('Dot product of u1 and u2: %.2f' % np.dot(u1, u2))

	# calculate logits for each event
	l1 = np.dot(x, u1) - offset
	l2 = np.dot(x, u2) - offset

	# calculate probability of each event
	p1 = sigmoid(l1)
	p2 = sigmoid(l2)

	# generate events
	e1 = bernoulli_draw(rs, p1)
	e2 = bernoulli_draw(rs, p2)

	if plot:

		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))

		ax[0].hist(l1, alpha=.5, bins=20, label='Event 1')
		ax[0].hist(l2, alpha=.5, bins=20, label='Event 2')
		ax[0].set_title('Event Logits')
		ax[0].legend()

		ax[1].hist(p1, alpha=.5, bins=20, label='Event 1')
		ax[1].hist(p2, alpha=.5, bins=20, label='Event 2')
		ax[1].set_title('Event Probabilities')
		ax[1].legend()

		plt.show()

	return x, e1, e2


def generate_vectors_by_similarity(rs, s):

	# generate vector 1
	u1 = normed_uniform(rs, N_FEATURES)

	# generate a second vector orthogonal to v1
	u1_ = normed_uniform(rs, N_FEATURES)
	u1_ = normalize(u1_ - u1 * np.dot(u1, u1_))

	# generate coefficients for event 2
	u2 = u1 * s + u1_ * (1 - s)

	return u1, u2


def find_offset(rs, logits, event_rate, step_size):

	offset = 0.
	rate = 1.

	while rate > event_rate:

		offset += step_size
		p = sigmoid(logits - offset)
		rate = np.mean(bernoulli_draw(rs, p))

	return offset


def normed_uniform(rs, n):
	return normalize(rs.rand(n) - .5)


def bernoulli_draw(rs, p):
	return (rs.rand(len(p)) < p).astype(int)


def glorot_uniform(rs, num_in, num_out):
	scale_factor = 2 * np.sqrt(6 / (num_in + num_out))
	return scale_factor * np.squeeze(rs.rand(num_in, num_out) - .5)


def logit(p):
	return np.log(p / (1 - p))


def sigmoid(l):
	return 1 / (1 + np.exp(-1 * l))


def normalize(v):
	return v / np.linalg.norm(v)


if __name__ == '__main__':
	main()