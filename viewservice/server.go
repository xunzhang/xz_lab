package viewservice

import "net"
import "net/rpc"
import "log"
import "time"
import "sync"
import "fmt"
import "os"
import "sync/atomic"

type ViewServer struct {
	mu       sync.Mutex
	l        net.Listener
	dead     int32 // for testing
	rpccount int32 // for testing
	me       string

	// Your declarations here.
  view View
  timestamp map[string]time.Time
  ACKed bool
}

//
// server Ping RPC handler.
//
func (vs *ViewServer) Ping(args *PingArgs, reply *PingReply) error {

	// Your code here.
  vs.mu.Lock()
  if vs.view.Primary != "" {
    Pdead := time.Since(vs.timestamp[vs.view.Primary]) > PingInterval * DeadPings
    if vs.ACKed == false && Pdead {
      vs.mu.Unlock()
      reply.View = vs.view
      return nil
    }
  }

  vs.timestamp[args.Me] = time.Now()
  if args.Me == vs.view.Primary && args.Viewnum == vs.view.Viewnum {
    vs.ACKed = true
  }
  if vs.view.Backup == "" {
    if vs.view.Primary == "" {
      // hotstart
      vs.view.Primary = args.Me
      vs.view.Viewnum++
    } else {
      if args.Me != vs.view.Primary {
       if vs.ACKed {
         vs.view.Backup = args.Me
         vs.view.Viewnum++
         vs.ACKed = false
       }
      } else {
        if args.Viewnum == 0 {
          vs.ACKed = false
        }
      }
    }
  } else {
    if args.Me == vs.view.Primary && args.Viewnum == 0 {
      vs.view.Primary, vs.view.Backup = vs.view.Backup, vs.view.Primary
      vs.view.Viewnum++
      vs.ACKed = false
    }
  }
  vs.mu.Unlock()

  reply.View = vs.view
  return nil
}

//
// server Get() RPC handler.
//
func (vs *ViewServer) Get(args *GetArgs, reply *GetReply) error {

	// Your code here.
  vs.mu.Lock()
  reply.View = vs.view
  vs.mu.Unlock()
	return nil
}

// for debug usage
func (vs *ViewServer) GetACKed(args *GetArgs, reply *GetACKedReply) error {
  reply.ACKed = vs.ACKed
  return nil
}

//
// tick() is called once per PingInterval; it should notice
// if servers have died or recovered, and change the view
// accordingly.
//
func (vs *ViewServer) tick() {
	// Your code here.
  if vs.view.Backup == "" || vs.ACKed == false { return }

  vs.mu.Lock()
  Bdead := time.Since(vs.timestamp[vs.view.Backup]) > PingInterval * DeadPings
  if vs.view.Primary == "" {
    if Bdead {
      vs.view.Backup = ""
      vs.view.Viewnum++
    } else {
      vs.view.Primary, vs.view.Backup = vs.view.Backup, ""
      vs.view.Viewnum++
      vs.ACKed = false
    }
  } else {
    Pdead := time.Since(vs.timestamp[vs.view.Primary]) > PingInterval * DeadPings
    if Pdead {
      if Bdead {
        vs.view.Primary, vs.view.Backup = "", ""
      } else {
        vs.view.Primary, vs.view.Backup = vs.view.Backup, ""
      }
      vs.view.Viewnum++
      vs.ACKed = false
    } else {
      if Bdead {
        vs.view.Backup = ""
        vs.view.Viewnum++
      }
    }
  }
  vs.mu.Unlock()
}

//
// tell the server to shut itself down.
// for testing.
// please don't change these two functions.
//
func (vs *ViewServer) Kill() {
	atomic.StoreInt32(&vs.dead, 1)
	vs.l.Close()
}

//
// has this server been asked to shut down?
//
func (vs *ViewServer) isdead() bool {
	return atomic.LoadInt32(&vs.dead) != 0
}

// please don't change this function.
func (vs *ViewServer) GetRPCCount() int32 {
	return atomic.LoadInt32(&vs.rpccount)
}

func StartServer(me string) *ViewServer {
	vs := new(ViewServer)
	vs.me = me

  // Your vs.* initializations here.
  vs.view = View{0, "", ""}
  vs.timestamp = make(map[string]time.Time)
  vs.ACKed = false

	// tell net/rpc about our RPC server and handlers.
	rpcs := rpc.NewServer()
	rpcs.Register(vs)

	// prepare to receive connections from clients.
	// change "unix" to "tcp" to use over a network.
	os.Remove(vs.me) // only needed for "unix"
	l, e := net.Listen("unix", vs.me)
	if e != nil {
		log.Fatal("listen error: ", e)
	}
	vs.l = l

	// please don't change any of the following code,
	// or do anything to subvert it.

	// create a thread to accept RPC connections from clients.
	go func() {
		for vs.isdead() == false {
			conn, err := vs.l.Accept()
			if err == nil && vs.isdead() == false {
				atomic.AddInt32(&vs.rpccount, 1)
				go rpcs.ServeConn(conn)
			} else if err == nil {
				conn.Close()
			}
			if err != nil && vs.isdead() == false {
				fmt.Printf("ViewServer(%v) accept: %v\n", me, err.Error())
				vs.Kill()
			}
		}
	}()

	// create a thread to call tick() periodically.
	go func() {
		for vs.isdead() == false {
			vs.tick()
			time.Sleep(PingInterval)
		}
	}()

	return vs
}
