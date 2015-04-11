stopWords = require('stopwords.lua')

for text in io.lines() do

    text = " " .. text:gsub("\\n", " "):gsub("\\\"\"", " ") .. " "

    text = text:gsub("http%S+", " url "):gsub("www%S+", " url ")
    
    text = text:gsub("(%a+)(%p+) ", "%1 %2 "):gsub(" (%p+)(%a+) ", " %1 %2 ")

    text = text:gsub(" [A-Z][A-Z]+ ", " intense %1")

    text = text:lower()

    text = text:gsub("can't", " not "):
                gsub("won't", " not "):
                gsub("n't ", " not "):
                gsub("'re ", " "):
                gsub("'ve ", " "):
                gsub("'ll ", " "):
                gsub("'d ", " "):
                gsub("'s ", " ")

    text = text:gsub("[8:=;]['`-]?[%]%)d]", " :) "):
                gsub("[8:=;]['`-]?[%[%(]", " :( "):
                gsub("[8:=;]['`-]?[p]", " :p "):
                gsub("[8:=;]['`-]?[|\\/]", " :| ") 

    text = text:gsub("$[.]?[%d]+[,.%d]*", " money ") 

    -- repeated punctuation 
    punct = {"!", "%?", "!%?", "%.", "%-"}
    for _, p in pairs(punct) do
        text = text:gsub("["..p.."]".."["..p.."]".."["..p.."]+", " repeat " .. p .. " ")
    end

    -- elongated words only do a subset of letter
    letters = {"m", "y", "w", "g", "h"}
    for _, l in pairs(letters) do
        prev = string.char(l:byte()-1)
        nxt = string.char(l:byte()+1)
        text = text:gsub(" ([a-"..prev..nxt.."-z]+)"..l..l..l.."+ ", " very %1"..l.." ")
    end
    text = text:gsub(" soo+ ", " very so ")
    text = text:gsub(" noo+ ", " very no ")

    for _, stopWord in pairs(stopWords) do
        text = text:gsub(" "..stopWord.." ", " ")
    end


    print(text)
end