#include <linux/module.h>
#include <linux/netlink.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/types.h>
#include <net/sock.h>

#define NETLINK_USER 31
#define MAX_PAYLOAD  256

static struct sock* nl_sk = NULL;
static int user_pid = -1;

/* simple header: u32 payload_len followed by raw bytes */
static void send_to_userspace(const void* data, u32 len) {
  struct sk_buff* skb_out;
  struct nlmsghdr* nlh;
  int res;

  if (user_pid <= 0) {
    pr_warn("netlink: no registered userspace pid yet\n");
    return;
  }

  /* allocate header + payload */
  skb_out = nlmsg_new(sizeof(u32) + len, GFP_KERNEL);
  if (!skb_out)
    return;

  nlh = nlmsg_put(skb_out, 0, 0, NLMSG_DONE, sizeof(u32) + len, 0);
  if (!nlh) {
    kfree_skb(skb_out);
    return;
  }

  /* write length then payload */
  *((u32*)nlmsg_data(nlh)) = len;
  memcpy(((u8*)nlmsg_data(nlh)) + sizeof(u32), data, len);

  res = nlmsg_unicast(nl_sk, skb_out, user_pid);
  if (res < 0)
    pr_err("netlink: failed to send message (res=%d)\n", res);
}

static void nl_recv_msg(struct sk_buff* skb) {
  char* msg = (char*)skb->data;
  int msg_len = skb->len;

  user_pid = NETLINK_CB(skb).portid;

  pr_info("netlink: registered userspace pid=%d, msg_len=%d\n", user_pid, msg_len);

  /*
   * For demo purposes, send a small payload back immediately after a
   * registration message. In real use you would trigger this from the
   * kernel producer path (e.g., hook, tracepoint, etc.).
   */
  if (msg_len >= 8 && !strncmp(msg, "REGISTER", 8)) {
    = static const u32 payload_ints[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    send_to_userspace(payload_ints, sizeof(payload_ints));
  }
}

static int __init infer_init(void) {
  struct netlink_kernel_cfg cfg = {
      .input = nl_recv_msg,
  };

  nl_sk = netlink_kernel_create(&init_net, NETLINK_USER, &cfg);
  if (!nl_sk) {
    pr_err("Failed to create netlink socket\n");
    return -ENOMEM;
  }

  pr_info("infer_nl: loaded\n");

  return 0;
}

static void __exit infer_exit(void) {
  if (nl_sk)
    netlink_kernel_release(nl_sk);
}

module_init(infer_init);
module_exit(infer_exit);
MODULE_LICENSE("GPL");
