package mapreduce

import "container/list"
import "fmt"

type WorkerInfo struct {
	address string
	// You can add definitions here.
}


// Clean up all workers by sending a Shutdown RPC to each one of them Collect
// the number of jobs each work has performed.
func (mr *MapReduce) KillWorkers() *list.List {
	l := list.New()
	for _, w := range mr.Workers {
		DPrintf("DoWork: shutdown %s\n", w.address)
		args := &ShutdownArgs{}
		var reply ShutdownReply
		ok := call(w.address, "Worker.Shutdown", args, &reply)
		if ok == false {
			fmt.Printf("DoWork: RPC %s shutdown error\n", w.address)
		} else {
			l.PushBack(reply.Njobs)
		}
	}
	return l
}

func (mr *MapReduce) RunMaster() *list.List {
	// Your code here
	mapDoneChannel := make(chan int, mr.nMap)
	reduceDoneChannel := make(chan int, mr.nReduce)

	doRpcJob := func (op JobType, worker string, id int) bool {
    args := DoJobArgs{mr.file, op, id, nReduce}
    if op == Reduce {
      args.NumOtherPhase = mr.nMap
    }
		var reply DoJobReply
		return call(worker, "Worker.DoJob", args, &reply)
	}

	// hand out map jobs
	for i := 0; i < mr.nMap; i++ {
		go func (id int) {
			for {
				var worker string
				var ok bool = false
				select {
				case worker = <- mr.registerChannel:
          mr.Workers[worker] = &WorkerInfo{worker}
					ok = doRpcJob(Map, worker, id)
				case worker = <- mr.freeChannel:
					ok = doRpcJob(Map, worker, id)
				}
				if ok {
					mapDoneChannel <- id
					mr.freeChannel <- worker
					return
				}
			}
		}(i)
	}
	// wait until all finished
	for i := 0; i < mr.nMap; i++ {
		<- mapDoneChannel
	}
	fmt.Println("All map jobs finished!")

	// hand out reduce jobs
	for i := 0; i < mr.nReduce; i++ {
		go func (id int) {
			for {
				var worker string
				var ok bool = false
				select {
				case worker = <- mr.registerChannel:
          mr.Workers[worker] = &WorkerInfo{worker}
					ok = doRpcJob(Reduce, worker, id)
				case worker = <- mr.freeChannel:
					ok = doRpcJob(Reduce, worker, id)
				}
				if ok {
					reduceDoneChannel <- id
					mr.freeChannel <- worker
					return
				}
			}
		}(i)
	}
	// wait until all finished
	for i := 0; i < mr.nReduce; i++ {
		<- reduceDoneChannel
	}
	fmt.Println("All reduce jobs finished!")

	return mr.KillWorkers()
}
