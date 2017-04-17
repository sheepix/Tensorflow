import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)


a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b


with tf.Session(config=config) as sess:
    print(sess.run(result))

sess = tf.Session(config=config)
with sess.as_default():
    print(result.eval())

print(result.eval(session=sess))



#交互式Session
sess = tf.InteractiveSession(config=config)
print(result.eval())
sess.close()
