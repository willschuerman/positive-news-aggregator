import scrapy

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://reasonstobecheerful.world/archives/']

    res = []

    def parse(self, response):
        for h3 in response.xpath('//h3/text()').getall():
            res.append(h3)

        # for next_page in response.css('a.next'):
        #     yield response.follow(next_page, self.parse)


    print(res)