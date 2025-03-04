FROM golang:1.20 as builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o scheduler ./cmd

FROM alpine:3.18
WORKDIR /root/
COPY --from=builder /app/scheduler .
CMD ["./scheduler"]

