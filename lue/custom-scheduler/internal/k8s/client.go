package k8s

import (
    "context"
    v1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
)

type Client struct {
    clientset *kubernetes.Clientset
}

func NewClient() (*Client, error) {
    config, err := rest.InClusterConfig()
    if err != nil {
        return nil, err
    }
    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        return nil, err
    }
    return &Client{clientset: clientset}, nil
}

func (c *Client) ListPendingPods(ctx context.Context) ([]*v1.Pod, error) {
    pods, err := c.clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
        FieldSelector: "status.phase=Pending",
    })
    if err != nil {
        return nil, err
    }
    var result []*v1.Pod
    for i := range pods.Items {
        result = append(result, &pods.Items[i])
    }
    return result, nil
}

func (c *Client) ListNodes(ctx context.Context) ([]*v1.Node, error) {
    nodes, err := c.clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
    if err != nil {
        return nil, err
    }
    var result []*v1.Node
    for i := range nodes.Items {
        result = append(result, &nodes.Items[i])
    }
    return result, nil
}

func (c *Client) BindPod(ctx context.Context, pod *v1.Pod, nodeName string) error {
    binding := &v1.Binding{
        ObjectMeta: metav1.ObjectMeta{
            Name:      pod.Name,
            Namespace: pod.Namespace,
            UID:       pod.UID,
        },
        Target: v1.ObjectReference{
            Kind: "Node",
            Name: nodeName,
        },
    }
    return c.clientset.CoreV1().Pods(pod.Namespace).Bind(ctx, binding)
}

