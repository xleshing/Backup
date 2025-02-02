package main

import (
	"fmt"
	"os"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	// "k8s.io/kubernetes/pkg/scheduler/internal/queue"
)

func main() {
	// 創建默認的 Scheduler
	sched, err := scheduler.New(
		scheduler.WithFrameworkOutOfTreeRegistry(framework.Registry{}),
	)
	if err != nil {
		klog.Fatalf("Failed to create scheduler: %v", err)
	}

	// 獲取 PriorityQueue
	priorityQueue := sched.SchedulingQueue()
	if pQueue, ok := priorityQueue.(*queue.PriorityQueue); ok {
		// 鎖定 activeQ 並獲取數據
		pQueue.Lock()
		defer pQueue.Unlock()

		// 遍歷 activeQ，打印 Pod 資訊
		for _, item := range pQueue.ActiveQ.Items {
			if podInfo, ok := item.(*framework.QueuedPodInfo); ok {
				fmt.Printf("Pod Name: %s, Namespace: %s\n", podInfo.Pod.Name, podInfo.Pod.Namespace)
			}
		}
	} else {
		klog.Error("Failed to cast SchedulingQueue to PriorityQueue")
		os.Exit(1)
	}
}
