package main

import (
    "context"
    "log"
    "os"
    "os/signal"
    "syscall"

    "custom-scheduler/internal/scheduler"
)

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    go func() {
        stop := make(chan os.Signal, 1)
        signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
        <-stop
        log.Println("Shutting down scheduler...")
        cancel()
    }()

    err := scheduler.Start(ctx)
    if err != nil {
        log.Fatalf("Scheduler exited with error: %v", err)
    }
}

