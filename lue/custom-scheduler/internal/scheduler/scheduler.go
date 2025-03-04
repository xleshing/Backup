package scheduler

import (
    "context"
    "time"
    "custom-scheduler/internal/k8s"
)

func Start(ctx context.Context) error {
    clientset, err := k8s.NewClient()
    if err != nil {
        return err
    }

    scheduler := NewBatchScheduler(clientset)
    go scheduler.Run(ctx, 5*time.Second)

    <-ctx.Done()
    return nil
}

