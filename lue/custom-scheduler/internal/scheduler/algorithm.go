package algorithm

import (
    v1 "k8s.io/api/core/v1"
)

type Assignment struct {
    Pod      *v1.Pod
    NodeName string
}

func RunBatchSchedule(pods []*v1.Pod, nodes []*v1.Node) []Assignment {
    var assignments []Assignment

    for i, pod := range pods {
        // 你自己的排程邏輯，這裡簡單輪詢示例
        node := nodes[i%len(nodes)]
        assignments = append(assignments, Assignment{
            Pod:      pod,
            NodeName: node.Name,
        })
    }

    return assignments
}

