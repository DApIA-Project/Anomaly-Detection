# try to import the mlviz module
# if installed 
USE_MLVIZ=True

if (USE_MLVIZ):
    try:
        USE_MLVIZ = False
        import MLviz
        USE_MLVIZ = True

        def setExperiment(experiments_name):
            MLviz.setExperiment(experiments_name)

        
        def startRun(run_name):
            lastRun = MLviz.getLastRun()
            print("lastRun", lastRun)
            if (lastRun == None):
                lastRun = "0-"
            lastRun = int(lastRun.split("-")[0])
            run = str(lastRun + 1) + "-" + run_name
            print("run", run)
            return MLviz.startRun(run)
        
        def log(name, v):
            return MLviz.log(name, v)
        
        def getLastRun():
            return MLviz.getLastRun()
    
    except:
        print("MLviz not installed")

if (not(USE_MLVIZ)):
    
    def setExperiment(experiments_name):
        return None
    
    def startRun(run_name):
        return None
    
    def log(name, v):
        return None
    
    def getLastRun():
        return ""
