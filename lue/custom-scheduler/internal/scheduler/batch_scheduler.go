package scheduler

import (
    "context"
    "time"
    "custom-scheduler/internal/k8s"
    "custom-scheduler/internal/scheduler/algorithm"
    "k8s.io/klog/v2"
    v1 "k8s.io/api/core/v1"
)

type BatchScheduler struct {
    client *k8s.Client
}

func NewBatchScheduler(client *k8s.Client) *BatchScheduler {
    return &BatchScheduler{client: client}
}

func (s *BatchScheduler) Run(ctx context.Context, interval time.Duration) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            s.schedule(ctx)
        case <-ctx.Done():
            return
        }
    }
}

func (s *BatchScheduler) schedule(ctx context.Context) {
    pods, err := s.client.ListPendingPods(ctx)
    if err != nil {
        klog.ErrorS(err, "Failed to list pending pods")
        return
    }

    if len(pods) == 0 {
        return
    }

    nodes, err := s.client.ListNodes(ctx)
    if err != nil {
        klog.ErrorS(err, "Failed to list nodes")
        return
    }

    assignments := algorithm.RunBatchSchedule(pods, nodes)

    for _, assign := range assignments {
        if err := s.client.BindPod(ctx, assign.Pod, assign.NodeName); err != nil {
            klog.ErrorS(err, "Failed to bind pod", "pod", assign.Pod.Name, "node", assign.NodeName)
        }
    }
}

